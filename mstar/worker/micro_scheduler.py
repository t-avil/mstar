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
class MixedBatchPlan:
    """Describes a mixed prefill+decode ("piggyback") step.

    Produced ONLY when ``MSTAR_MIXED_WALK`` is enabled. Attached to a
    ``ScheduledBatch`` whose primary ``graph_walk`` is the latency-sensitive
    decode walk; the named prefill request(s) ride the same scheduler step so
    a freshly-arrived request's prefill no longer waits a full cycle behind
    the running decode batch (continuous batching, vLLM-v1 style).

    The decode and prefill requests share one ``node_name`` (in M*'s
    Qwen3-Omni topology the Thinker decode and all Thinker prefill walks run on
    the same "Thinker" node), so a single mixed varlen forward can serve both.
    See DESIGN_mixed_walk.md for the forward/replay contract this descriptor
    drives.
    """
    decode_rids: list[str]
    prefill_rids: list[str]
    decode_walk: str
    prefill_walk: str | None
    token_budget: int
    prefill_chunk_cap: int


@dataclass
class ScheduledBatch:
    """A batch of nodes ready to be executed."""
    node_name: str
    graph_walk: str
    node_objects: dict[str,GraphNode]
    # request_id -> worker_graph_id (for push-back on OOM)
    request_to_worker_graph: dict[str, str] = None
    # Set ONLY under MSTAR_MIXED_WALK when a waiting prefill piggybacks onto
    # this decode batch. None on every default-path batch, so the flag-OFF
    # behavior (and every consumer that doesn't inspect it) is unchanged.
    mixed_plan: "MixedBatchPlan | None" = None


# Priority: lower value = higher priority
# KV-cache decode is most latency-sensitive
PRIORITY = {
    EngineType.KV_CACHE: 0,
    EngineType.STATELESS: 2,
}

class SchedulingType(Enum):
    PRIORITY = "priority"
    ROUND_ROBIN = "round_robin"


def is_decode_walk(graph_walk: str) -> bool:
    """Heuristic split of decode (1-token AR step) walks from prefill walks.

    A walk is a decode if its name is exactly ``decode`` or ends in
    ``_decode`` (M*'s ``thinker_decode`` / ``talker_decode``). M*'s prefill
    walks carry a ``prefill`` segment (``prefill_text`` / ``prefill_audio`` /
    ``prefill_vision`` / ``talker_prefill``), so this cleanly identifies the
    latency-sensitive decode that should remain the PRIMARY of a mixed step,
    with prefills piggybacking onto it. Used only on the env-gated mixed path;
    rationale and override points are in DESIGN_mixed_walk.md.
    """
    return graph_walk == "decode" or graph_walk.endswith("_decode")


def plan_mixed_budget(
    decode_count: int,
    prefill_candidates: list,
    token_budget: int,
    prefill_chunk_cap: int,
    max_prefill_requests: int = 1,
    token_count_fn=None,
) -> list[int]:
    """Pure token-budget admission for one mixed prefill+decode step.

    Mirrors vLLM v1's single-token-budget loop (scheduler.py: running decodes
    consume one token each, then waiting prefills consume the remaining
    budget). Returns the indices INTO ``prefill_candidates`` admitted to
    piggyback this step (highest-priority/scan-order first).

    ``token_count_fn(candidate) -> int`` estimates a candidate's prefill token
    cost; it defaults to the conservative ``prefill_chunk_cap`` because exact
    per-request prefill lengths are not reliably available at schedule time
    (they live in the request's pending input tensors, not in the ready-node
    metadata the scheduler scans). Every candidate's cost is capped at
    ``prefill_chunk_cap`` so the captured-graph shape set stays finite — see
    the capture-shape risk section of DESIGN_mixed_walk.md.

    Pure function (no scheduler/queue state) so it is unit-testable on CPU.
    """
    if token_count_fn is None:
        token_count_fn = lambda _c: prefill_chunk_cap  # noqa: E731
    admitted: list[int] = []
    used = decode_count  # each running decode contributes exactly 1 query token
    for idx, cand in enumerate(prefill_candidates):
        if len(admitted) >= max_prefill_requests:
            break
        cost = min(token_count_fn(cand), prefill_chunk_cap)
        if used + cost > token_budget:
            continue
        admitted.append(idx)
        used += cost
    return admitted


class MicroScheduler:
    """
    Simple MVP scheduler: scans all worker graph queues for ready nodes,
    groups by node name, returns the highest-priority group.
    """

    # Seconds to wait before retrying a held request after OOM
    HOLD_BACKOFF_SECONDS = 0.05

    def __init__(
        self, engine_manager: EngineManager,
        sched_type=SchedulingType.ROUND_ROBIN,
        tp_rank_zero_nodes: set[str] | None = None,
        max_consec_tp_follower_batches: int = 1,
    ):
        self.engine_manager = engine_manager
        self.batch_number = 0
        self.sched_type = sched_type

        # tensor parallel
        self.tp_rank_zero_nodes = tp_rank_zero_nodes
        self.tp_batches_pending_schedule = deque()
        self.num_consec_tp_follower_batches = 0
        self.max_consec_tp_follower_batches = max_consec_tp_follower_batches

        self.node_and_walk_to_last_batch_num = {}
        # request_id -> monotonic time until which the request is held
        self.held_until: dict[str, float] = {}

        # --- MSTAR_MIXED_WALK: continuous-batching (prefill piggybacks decode) ---
        # Default OFF -> strict one-(node, graph_walk)-per-step, byte-identical to
        # the pre-existing scheduler. When ON, get_next_batch may admit one (or a
        # few) waiting prefill requests onto the running decode batch under a
        # shared token budget. See DESIGN_mixed_walk.md.
        self.mixed_walk_enabled = os.environ.get("MSTAR_MIXED_WALK", "0") == "1"
        self.mixed_token_budget = int(
            os.environ.get("MSTAR_MIXED_TOKEN_BUDGET", "8192")
        )
        self.mixed_prefill_chunk_cap = int(
            os.environ.get("MSTAR_MIXED_PREFILL_CHUNK", "512")
        )
        self.mixed_max_prefill_requests = int(
            os.environ.get("MSTAR_MIXED_MAX_PREFILL_REQS", "1")
        )
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
        # Collect all ready (node_name, request_id, graph_walk) tuples
        # grouped by node name
        node_name_to_requests: dict[str, list[ReadyNodeEntry]] = {}
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

        for worker_graph_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if request_id not in worker_graphs_manager.per_request_info:
                    continue  # request was removed between scheduling cycles
                if request_id in self.pending_removes:
                    continue  # remove deferred for in-flight safety; don't start new work
                # Skip requests in OOM backoff
                if request_id in self.held_until:
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

        mixed_plan = self._maybe_plan_mixed(
            best_node_name=best_node_name,
            graph_walk=graph_walk,
            node_name_to_requests=node_name_to_requests,
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
            worker_graphs_manager=worker_graphs_manager,
        )

        return ScheduledBatch(
            node_name=best_node_name,
            graph_walk=graph_walk,
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
            mixed_plan=mixed_plan,
        )

    def _engine_is_kv_cache(self, node_name: str) -> bool:
        if node_name not in self.engine_manager.node_to_engine:
            return False
        return self.engine_manager.get_engine(node_name).engine_type() == EngineType.KV_CACHE

    def _maybe_plan_mixed(
        self,
        best_node_name: str,
        graph_walk: str,
        node_name_to_requests: dict[str, list[ReadyNodeEntry]],
        node_objects: dict,
        request_to_worker_graph: dict,
        worker_graphs_manager: WorkerGraphsManager,
    ) -> "MixedBatchPlan | None":
        """Piggyback waiting prefill(s) onto a decode batch (MSTAR_MIXED_WALK).

        No-op (returns None) unless the flag is on AND the just-selected
        primary is a latency-sensitive decode on a KV-cache node. When it
        applies, it admits prefill request(s) of OTHER walks on the SAME node
        under the shared token budget, pops their ready nodes into
        ``node_objects`` (mutated in place, like the decode loop above), and
        returns the plan describing the mixed step. On the flag-OFF path this
        method is never called with effect — every batch keeps ``mixed_plan=None``.
        """
        if not self.mixed_walk_enabled:
            return None
        if not node_objects:
            return None
        if not is_decode_walk(graph_walk):
            return None  # only piggyback ONTO a decode primary, never onto a prefill
        if not self._engine_is_kv_cache(best_node_name):
            return None

        decode_rids = list(node_objects.keys())
        prefill_candidates = [
            e for e in node_name_to_requests[best_node_name]
            if e.request_id not in node_objects and not is_decode_walk(e.graph_walk)
        ]
        if not prefill_candidates:
            return None

        admitted = plan_mixed_budget(
            decode_count=len(decode_rids),
            prefill_candidates=prefill_candidates,
            token_budget=self.mixed_token_budget,
            prefill_chunk_cap=self.mixed_prefill_chunk_cap,
            max_prefill_requests=self.mixed_max_prefill_requests,
        )
        if not admitted:
            return None

        prefill_rids: list[str] = []
        prefill_walk: str | None = None
        for idx in admitted:
            entry = prefill_candidates[idx]
            queue = worker_graphs_manager.queues[entry.worker_graph_id]
            popped = queue.pop_ready_nodes(entry.request_id, [best_node_name])
            if not popped:
                continue
            assert len(popped) == 1
            node_objects[entry.request_id] = popped[0]
            request_to_worker_graph[entry.request_id] = entry.worker_graph_id
            prefill_rids.append(entry.request_id)
            prefill_walk = entry.graph_walk
            self.node_and_walk_to_last_batch_num[
                (best_node_name, entry.graph_walk)
            ] = self.batch_number

        if not prefill_rids:
            return None

        logger.debug(
            "MicroScheduler MIXED step on node %s: decode walk %s (%d reqs) + "
            "prefill walk %s (%d reqs)",
            best_node_name, graph_walk, len(decode_rids),
            prefill_walk, len(prefill_rids),
        )
        return MixedBatchPlan(
            decode_rids=decode_rids,
            prefill_rids=prefill_rids,
            decode_walk=graph_walk,
            prefill_walk=prefill_walk,
            token_budget=self.mixed_token_budget,
            prefill_chunk_cap=self.mixed_prefill_chunk_cap,
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
