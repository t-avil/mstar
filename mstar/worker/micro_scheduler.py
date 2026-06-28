import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from mstar.engine.base import EngineType
from mstar.graph.base import GraphNode
from mstar.utils.ipc_format import ScheduleTPNode
from mstar.worker.engine_manager import EngineManager
from mstar.worker.node_manager_utils import WorkerGraphsManager

logger = logging.getLogger(__name__)


@dataclass
class ReadyNodeEntry:
    """A ready node entry for a single request."""
    request_id: str
    worker_graph_id: str
    graph_walk: str


@dataclass
class ScheduledBatch:
    """A batch of nodes ready to be executed."""
    node_name: str
    graph_walk: str
    node_objects: dict[str,GraphNode]
    # request_id -> worker_graph_id (for push-back on OOM)
    request_to_worker_graph: dict[str, str] = None


# Priority: lower value = higher priority
# KV-cache decode is most latency-sensitive
PRIORITY = {
    EngineType.KV_CACHE: 0,
    EngineType.STATELESS: 2,
}

class SchedulingType(Enum):
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"


class MicroScheduler:
    """
    Simple MVP scheduler: scans all worker graph queues for ready nodes,
    groups by node name, returns the highest-priority group.
    """

    # Seconds to wait before retrying a held request after OOM
    HOLD_BACKOFF_SECONDS = 0.05

    # Poll granularity used inside the encoder coalescing window. Re-scan the
    # ready queues a few times across the wait window rather than busy-spinning.
    COALESCE_POLL_SECONDS = 0.0005

    # Node names whose forward is worth coalescing across requests. These are
    # the STATELESS multimodal encoders (Qwen3-Omni prefill_audio /
    # prefill_vision walks); their submodules implement forward_batched.
    ENCODER_NODE_NAMES = frozenset({"audio_encoder", "vision_encoder"})

    def __init__(
        self, engine_manager: EngineManager,
        sched_type=SchedulingType.ROUND_ROBIN,
        tp_rank_zero_nodes: set[str] | None = None,
        max_consec_tp_follower_batches: int = 1,
    ):
        self.engine_manager = engine_manager
        self.batch_number = 0
        self.sched_type = sched_type

        # --- Cross-request encoder coalescing (env-gated, default OFF) ---
        # When enabled, once an encoder node is the selected node for this
        # scheduling cycle, briefly wait to accumulate more ready encoder
        # requests for the SAME (node, graph_walk) and dispatch them as one
        # forward_batched call. Only encoder nodes get a window; decode and
        # other latency-critical nodes are never delayed. The window aborts
        # early once it is full, the wait elapses, OR a higher-priority
        # (KV-cache decode) node becomes ready.
        self._coalesce_enabled = os.environ.get(
            "MSTAR_ENCODER_COALESCE", "0"
        ) in ("1", "true", "True")
        try:
            self._coalesce_wait_s = max(
                0.0,
                float(os.environ.get("MSTAR_ENCODER_COALESCE_WAIT_MS", "5")) / 1000.0,
            )
        except ValueError:
            self._coalesce_wait_s = 0.005
        try:
            self._coalesce_max_batch = max(
                1, int(os.environ.get("MSTAR_ENCODER_COALESCE_MAX_BATCH", "32"))
            )
        except ValueError:
            self._coalesce_max_batch = 32
        if self._coalesce_enabled:
            logger.info(
                "MicroScheduler: encoder coalescing ON (wait=%.1fms max_batch=%d nodes=%s)",
                self._coalesce_wait_s * 1000.0,
                self._coalesce_max_batch,
                sorted(self.ENCODER_NODE_NAMES),
            )

        # tensor parallel
        self.tp_rank_zero_nodes = tp_rank_zero_nodes
        self.tp_batches_pending_schedule = deque()
        self.num_consec_tp_follower_batches = 0
        self.max_consec_tp_follower_batches = max_consec_tp_follower_batches

        self.node_and_walk_to_last_batch_num = {}
        # request_id -> monotonic time until which the request is held
        self.held_until: dict[str, float] = {}
        # Rids with a deferred remove; stop initiating new work for them.
        # Shared by reference with Worker._pending_removes.
        self.pending_removes: set[str] = set()

    def _select_node_priority(
        self, node_name_to_requests: dict[str, list[ReadyNodeEntry]]
    ):
        # Pick the node name with highest priority (lowest PRIORITY value)
        best_node_name = None
        best_priority = float("inf")

        for node_name in node_name_to_requests:
            if node_name not in self.engine_manager.node_to_engine:
                continue
            engine = self.engine_manager.get_engine(node_name)
            prio = PRIORITY.get(engine.engine_type(), 99)
            if prio < best_priority:
                best_priority = prio
                best_node_name = node_name
        if best_node_name is None:
            return None, None
        entries = node_name_to_requests[best_node_name]

        # Enforce same graph_walk for the entire batch.
        # Pick the most common graph_walk to maximize batch size;
        # remaining requests stay in the queue for the next cycle.
        walk_counts: dict[str, int] = {}
        for e in entries:
            walk_counts[e.graph_walk] = walk_counts.get(e.graph_walk, 0) + 1
        graph_walk = max(walk_counts, key=walk_counts.get)

        return node_name, graph_walk

    def _select_node_rr(
        self, node_name_to_requests: dict[str, list[ReadyNodeEntry]]
    ):
        best_node_name = None
        best_graph_walk = None
        least_recent_step = float('inf')

        for node_name, reqs in node_name_to_requests.items():
            for req in reqs:
                step = self.node_and_walk_to_last_batch_num.get((
                    node_name, req.graph_walk
                ), 0)
                if step < least_recent_step:
                    least_recent_step = step
                    best_node_name = node_name
                    best_graph_walk = req.graph_walk
        return best_node_name, best_graph_walk

    def hold_requests(self, request_ids: list[str]) -> None:
        """Put requests on hold for a brief backoff period after OOM."""
        deadline = time.monotonic() + self.HOLD_BACKOFF_SECONDS
        for rid in request_ids:
            self.held_until[rid] = deadline

    def register_tp_follow(
        self, message: ScheduleTPNode
    ):
        self.tp_batches_pending_schedule.append(message)

    def _try_schedule_tp_follow(
        self, worker_graphs_manager: WorkerGraphsManager,
    ) -> ScheduledBatch | None:
        if len(self.tp_batches_pending_schedule) == 0:
            return
        first_tp_node: ScheduleTPNode = self.tp_batches_pending_schedule[0]
        if self.num_consec_tp_follower_batches >= self.max_consec_tp_follower_batches and \
                self.has_ready_excluding(
                    worker_graphs_manager,
                    (first_tp_node.node_name, first_tp_node.graph_walk)
                ):
            return
        # check if batch is ready
        node_partition = worker_graphs_manager.get_partition_for_node(first_tp_node.node_name)
        wgid = worker_graphs_manager.get_worker_graph_id_for_node(
            first_tp_node.request_ids[0], first_tp_node.node_name
        )
        queue = worker_graphs_manager.queues[wgid]
        for rid in first_tp_node.request_ids:
            wg = queue.per_request_queues[rid]
            if first_tp_node.node_name not in wg.ready_node_names:
                return
            fwd_info = worker_graphs_manager.get_fwd_info(rid, node_partition)
            # check if the node is ready on the engine level
            # (e.g., for AR, whether the kv cache is read in)
            engine = self.engine_manager.get_engine(first_tp_node.node_name)
            if not engine.check_ready(first_tp_node.node_name, rid, fwd_info):
                return

        node_objects = {}
        request_to_worker_graph = {}

        # TODO: this code is also repeated below, should pull into a helper fn
        for rid in first_tp_node.request_ids:
            popped = queue.pop_ready_nodes(rid, [first_tp_node.node_name])
            if popped:
                assert len(popped) == 1
                node_objects[rid] = popped[0]
                request_to_worker_graph[rid] = wgid

        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[(
            first_tp_node.node_name, first_tp_node.graph_walk
        )] = self.batch_number

        self.tp_batches_pending_schedule.popleft()

        return ScheduledBatch(
            node_name=first_tp_node.node_name,
            graph_walk=first_tp_node.graph_walk,
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
        )


    def _collect_ready_nodes(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        target_node_name: str | None = None,
        target_graph_walk: str | None = None,
        exclude_target: tuple[str, str] | None = None,
    ) -> dict[str, list[ReadyNodeEntry]]:
        """Scan every worker-graph queue for ready, engine-ready nodes that
        rank 0 may initiate, grouped by node name. Mirrors the (node, walk,
        request) filtering used by get_next_batch. Does not pop or mutate
        queue state. Safe to call repeatedly (used by the coalescing window).
        """
        now = time.monotonic()
        node_name_to_requests: dict[str, list[ReadyNodeEntry]] = {}
        for worker_graph_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if request_id not in worker_graphs_manager.per_request_info:
                    continue  # request was removed between scheduling cycles
                if request_id in self.pending_removes:
                    continue  # remove deferred for in-flight safety; don't start new work
                # Skip requests in OOM backoff
                if request_id in self.held_until and self.held_until[request_id] > now:
                    continue
                for sname in node_names:
                    if sname not in self.tp_rank_zero_nodes:
                        continue # only rank 0 can initiate scheduling!
                    if target_node_name is not None and sname != target_node_name:
                        continue
                    node_partition = worker_graphs_manager.get_partition_for_node(sname)
                    graph_walk = worker_graphs_manager.get_graph_walk(request_id, node_partition)
                    if target_graph_walk is not None and graph_walk != target_graph_walk:
                        continue
                    if exclude_target is not None and (sname, graph_walk) == exclude_target:
                        continue
                    fwd_info = worker_graphs_manager.get_fwd_info(request_id, node_partition)
                    # check if the node is ready on the engine level
                    # (e.g., for AR, whether the kv cache is read in)
                    engine = self.engine_manager.get_engine(sname)
                    if not engine.check_ready(sname, request_id, fwd_info):
                        continue
                    node_name_to_requests.setdefault(sname, []).append(
                        ReadyNodeEntry(request_id, worker_graph_id, graph_walk)
                    )
        return node_name_to_requests

    def _has_higher_priority_ready(
        self,
        node_name_to_requests: dict[str, list[ReadyNodeEntry]],
        encoder_node_name: str,
    ) -> bool:
        """True if any ready node has strictly higher priority (lower PRIORITY
        value) than the encoder node — e.g. a KV-cache decode step. Used to
        abort the coalescing window early so latency-critical work is never
        starved by the encoder wait.
        """
        if encoder_node_name not in self.engine_manager.node_to_engine:
            return False
        enc_engine = self.engine_manager.get_engine(encoder_node_name)
        enc_prio = PRIORITY.get(enc_engine.engine_type(), 99)
        for sname in node_name_to_requests:
            if sname == encoder_node_name:
                continue
            if sname not in self.engine_manager.node_to_engine:
                continue
            engine = self.engine_manager.get_engine(sname)
            if PRIORITY.get(engine.engine_type(), 99) < enc_prio:
                return True
        return False

    def _coalesce_encoder_window(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        node_name: str,
        graph_walk: str,
        max_batch_size: int | None,
        target_node_name: str | None,
        target_graph_walk: str | None,
        exclude_target: tuple[str, str] | None,
        initial_entries: list[ReadyNodeEntry],
    ) -> list[ReadyNodeEntry]:
        """Bounded wait window that accumulates more ready encoder requests for
        (node_name, graph_walk) so they dispatch as one forward_batched call.

        Terminates as soon as ANY of these holds (all bound the added delay):
          * the batch reaches the coalescing cap (or the caller's
            max_batch_size, whichever is smaller),
          * MSTAR_ENCODER_COALESCE_WAIT_MS has elapsed,
          * a higher-priority (decode) node becomes ready.

        Returns the (possibly larger) list of entries to dispatch.
        """
        cap = self._coalesce_max_batch
        if max_batch_size is not None:
            cap = min(cap, max_batch_size)

        entries = initial_entries
        if len(entries) >= cap:
            return entries

        deadline = time.monotonic() + self._coalesce_wait_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(self.COALESCE_POLL_SECONDS, remaining))
            node_name_to_requests = self._collect_ready_nodes(
                worker_graphs_manager,
                target_node_name=target_node_name,
                target_graph_walk=target_graph_walk,
                exclude_target=exclude_target,
            )
            entries = [
                e for e in node_name_to_requests.get(node_name, [])
                if e.graph_walk == graph_walk
            ]
            if len(entries) >= cap:
                break
            # Don't starve latency-critical decode: if a higher-priority node
            # became ready during the wait, stop accumulating and dispatch now.
            if self._has_higher_priority_ready(node_name_to_requests, node_name):
                break

        if entries:
            logger.debug(
                "MicroScheduler coalesced %d encoder requests for node %s walk %s",
                len(entries), node_name, graph_walk,
            )
        return entries

    def get_next_batch(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        max_batch_size: int | None = None,
        target_node_name: str | None = None,
        target_graph_walk: str | None = None,
        exclude_target: tuple[str, str] | None = None,
    ) -> ScheduledBatch | None:
        """
        Scans all worker graph queues for ready nodes.
        Groups by node name. Returns highest-priority group.

        Args:
            max_batch_size: If set, limit the number of requests in the batch.
                Useful for CUDA graph compatibility (must match captured sizes).
            target_node_name: If set, only schedule this node name.
            target_graph_walk: If set, only schedule this graph walk.
            exclude_target: If set, skip this (node_name, graph_walk) pair.
        """
        now = time.monotonic()

        # Expire stale hold entries
        self.held_until = {
            rid: t for rid, t in self.held_until.items() if t > now
        }

        tp_follow_batch = self._try_schedule_tp_follow(worker_graphs_manager)
        if tp_follow_batch is None:
            self.num_consec_tp_follower_batches = 0
        else:
            self.num_consec_tp_follower_batches += 1
            return tp_follow_batch

        node_name_to_requests = self._collect_ready_nodes(
            worker_graphs_manager,
            target_node_name=target_node_name,
            target_graph_walk=target_graph_walk,
            exclude_target=exclude_target,
        )

        if not node_name_to_requests:
            return None

        if self.sched_type == SchedulingType.PRIORITY:
            best_node_name, graph_walk = self._select_node_priority(node_name_to_requests)
        elif self.sched_type == SchedulingType.ROUND_ROBIN:
            best_node_name, graph_walk = self._select_node_rr(node_name_to_requests)
        else:
            raise NotImplementedError(f"Unkown scheduling type {self.sched_type}")

        if best_node_name is None:
            return None

        # Pop ready nodes for all requests of this node name
        entries = [e for e in node_name_to_requests[best_node_name] \
                   if e.graph_walk == graph_walk]

        # Cross-request encoder coalescing window. Only when enabled and the
        # selected node is a (STATELESS) multimodal encoder: briefly wait to
        # accumulate more ready requests for the SAME (node, walk) so they run
        # as one forward_batched call. Decode / non-encoder nodes are never
        # delayed (they were already selected and dispatched immediately).
        if (
            self._coalesce_enabled
            and best_node_name in self.ENCODER_NODE_NAMES
            and len(entries) < self._coalesce_max_batch
        ):
            entries = self._coalesce_encoder_window(
                worker_graphs_manager,
                best_node_name,
                graph_walk,
                max_batch_size=max_batch_size,
                target_node_name=target_node_name,
                target_graph_walk=target_graph_walk,
                exclude_target=exclude_target,
                initial_entries=entries,
            )

        # Limit batch size if requested (e.g., for CUDA graph compatibility)
        if max_batch_size is not None and len(entries) > max_batch_size:
            entries = entries[:max_batch_size]

        node_objects = {}
        request_to_worker_graph = {}

        for entry in entries:
            queue = worker_graphs_manager.queues[entry.worker_graph_id]
            popped = queue.pop_ready_nodes(entry.request_id, [best_node_name])
            if popped:
                assert len(popped) == 1
                node_objects[entry.request_id] = popped[0]
                request_to_worker_graph[entry.request_id] = entry.worker_graph_id

        if not node_objects:
            return None

        logger.debug(
            "MicroScheduler scheduling node %s with graph walk %s for %d requests",
            best_node_name, graph_walk, len(node_objects)
        )
        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[(
            best_node_name, graph_walk
        )] = self.batch_number

        return ScheduledBatch(
            node_name=best_node_name,
            graph_walk=graph_walk,
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
        )

    def has_ready_excluding(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        exclude_target: tuple[str, str] | None,
    ) -> bool:
        """Cheap peek: any worker-graph queue ready with a (node, walk) other
        than `exclude_target`? Used by the speculation path to decide whether
        breaking the spec chain for fairness is actually warranted on this
        worker — on single-walk workers (e.g. Orpheus LLM) the answer is
        always False, so speculation can run every iter.

        Does NOT pop or modify queue state. Mirrors the ready-scan in
        get_next_batch but stops at the first match.
        """
        now = time.monotonic()
        # Don't bother expiring held_until here — we only read it; the next
        # get_next_batch call will refresh.
        for _worker_graph_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if request_id not in worker_graphs_manager.per_request_info:
                    continue
                if request_id in self.held_until and self.held_until[request_id] > now:
                    continue
                for sname in node_names:
                    node_partition = worker_graphs_manager.get_partition_for_node(sname)
                    graph_walk = worker_graphs_manager.get_graph_walk(
                        request_id, node_partition,
                    )
                    if exclude_target is not None and (sname, graph_walk) == exclude_target:
                        continue
                    fwd_info = worker_graphs_manager.get_fwd_info(request_id, node_partition)
                    engine = self.engine_manager.get_engine(sname)
                    if not engine.check_ready(sname, request_id, fwd_info):
                        continue
                    return True
        return False
