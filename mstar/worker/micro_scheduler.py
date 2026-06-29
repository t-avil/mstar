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

# ---------------------------------------------------------------------------
# MSTAR_ENCODER_ASYNC: pipeline the vision/audio encoder ahead of the Thinker.
#
# Default OFF. When ON, the micro-scheduler treats ``vision_encoder`` and
# ``audio_encoder`` (both STATELESS) as higher-priority than the Thinker
# decode/prefill — but only until ``MSTAR_ENCODER_ASYNC_DEPTH`` encoded
# buffers are in flight (i.e. produced by the encoder but not yet consumed
# by the Thinker's matching ``prefill_vision`` / ``prefill_audio`` step).
#
# Without this flag, the encoder for request N+1 only runs after request N
# has been dispatched to the Thinker. The encoder GPU sits idle during the
# Thinker decode of N. With this flag, the encoder for N+1 starts while N
# is in ``thinker_decode``, so when the Thinker is ready for N+1's prefill
# the encoded buffer is already populated — zero encoder wait.
#
# Bounding the in-flight depth prevents the encoder from monopolizing the
# GPU under heavy admission and is the K=4 ceiling from the experiment spec.
# ---------------------------------------------------------------------------
def _encoder_async_enabled() -> bool:
    # Default OFF. The full I2T sweep showed PROMISING at B>=16, but the feature
    # has been flaky in integration testing, so it is opt-in via
    # MSTAR_ENCODER_ASYNC=1. When enabled, pipelining is still restricted to the
    # vision encoder only (audio is too cheap to amortize the speculation cost;
    # see LEARNINGS §9.2/§9.5).
    return os.environ.get("MSTAR_ENCODER_ASYNC", "0") in ("1", "true", "True")


def _encoder_async_depth() -> int:
    raw = os.environ.get("MSTAR_ENCODER_ASYNC_DEPTH", "4")
    try:
        v = int(raw)
        return v if v > 0 else 4
    except ValueError:
        return 4


def _encoder_async_node_names() -> frozenset[str]:
    """Which encoder nodes are eligible for async pipelining.

    Default: vision only. Audio encoder is too cheap (~10-20ms vs vision
    ~160ms) for speculation to pay off and the S2T full sweep showed -18%
    req/s at B=16+ when audio was included.

    Override with MSTAR_ENCODER_ASYNC_PATHS=vision,audio to opt audio back
    in (or =none to disable both even when MSTAR_ENCODER_ASYNC=1).
    """
    raw = os.environ.get("MSTAR_ENCODER_ASYNC_PATHS", "vision").strip().lower()
    if raw in ("", "none", "off"):
        return frozenset()
    parts = {p.strip() for p in raw.split(",") if p.strip()}
    out = set()
    if "vision" in parts:
        out.add("vision_encoder")
    if "audio" in parts:
        out.add("audio_encoder")
    return frozenset(out)


# Node names the encoder-async path treats as "encoder" walks.
# Resolved at module-load; processes inheriting env vars from the server
# launcher get the right set without restart-juggling per request.
_ENCODER_NODE_NAMES = _encoder_async_node_names()
# Graph walks whose first node consumes an encoder output on the Thinker.
_ENCODER_CONSUMING_WALKS = frozenset({"prefill_vision", "prefill_audio"})


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
        # Rids with a deferred remove; stop initiating new work for them.
        # Shared by reference with Worker._pending_removes.
        self.pending_removes: set[str] = set()

        # --- MSTAR_ENCODER_ASYNC bookkeeping ---------------------------------
        # ``encoder_async_enabled``: cached at construction so a single feature-
        # flag check governs every dispatch decision. Reading the env var once
        # also keeps the hot path branch cheap.
        # ``encoder_async_depth``: max number of encoded-but-not-yet-Thinker-
        # consumed buffers we'll let pile up. Each batched encoder dispatch
        # counts as one "in flight" credit regardless of how many requests
        # were coalesced into the batch — batching is independent of
        # pipelining depth, and the depth bound is about wall-clock head-room
        # (how far ahead of Thinker the encoder is allowed to run), not
        # about KV cache memory.
        # ``encoder_async_in_flight``: incremented when the scheduler returns
        # an encoder batch; decremented when the matching ``prefill_vision``
        # / ``prefill_audio`` step is scheduled on the Thinker (the
        # downstream consumer of the buffered embeddings).
        self.encoder_async_enabled = _encoder_async_enabled()
        self.encoder_async_depth = _encoder_async_depth()
        self.encoder_async_in_flight = 0
        if self.encoder_async_enabled:
            logger.info(
                "MicroScheduler: MSTAR_ENCODER_ASYNC=1 (depth=%d). "
                "Encoder walks pipeline ahead of Thinker decode.",
                self.encoder_async_depth,
            )

    def _maybe_pick_async_encoder(
        self, node_name_to_requests: dict[str, list[ReadyNodeEntry]]
    ) -> tuple[str | None, str | None]:
        """If MSTAR_ENCODER_ASYNC is enabled, prefer a ready encoder node.

        Returns ``(node_name, graph_walk)`` for the encoder pick, or
        ``(None, None)`` if no preemption applies (flag off, no encoder
        ready, depth saturated, or no encoder node has a ready request).

        Depth budget: when ``encoder_async_in_flight`` reaches
        ``encoder_async_depth`` we fall back to normal scheduling so the
        Thinker can catch up. This prevents the encoder from running an
        unbounded number of buffers ahead — bad both for GPU memory
        (each buffer pins vision/audio embeds) and for first-token latency
        of the requests already in the Thinker.
        """
        if not self.encoder_async_enabled:
            return None, None
        if self.encoder_async_in_flight >= self.encoder_async_depth:
            return None, None
        for node_name in _ENCODER_NODE_NAMES:
            entries = node_name_to_requests.get(node_name)
            if not entries:
                continue
            # Bias toward whichever walk is least-recently scheduled, mirroring
            # the round-robin tie-breaker for non-encoder nodes. In practice
            # both ``audio_encoder`` and ``vision_encoder`` only emit a single
            # walk (``prefill_audio`` / ``prefill_vision``) so this collapses
            # to the first entry's walk, but we keep the RR semantics for
            # robustness if a future walk reuses these encoder nodes.
            walk_counts: dict[str, int] = {}
            for e in entries:
                walk_counts[e.graph_walk] = walk_counts.get(e.graph_walk, 0) + 1
            graph_walk = max(walk_counts, key=walk_counts.get)
            return node_name, graph_walk
        return None, None

    def _select_node_priority(
        self, node_name_to_requests: dict[str, list[ReadyNodeEntry]]
    ):
        # MSTAR_ENCODER_ASYNC: pipeline the encoder ahead of the Thinker when
        # there's budget. See ``_maybe_pick_async_encoder`` for the depth
        # bound and rationale.
        async_node, async_walk = self._maybe_pick_async_encoder(
            node_name_to_requests
        )
        if async_node is not None:
            return async_node, async_walk

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
        # MSTAR_ENCODER_ASYNC: short-circuit the RR sweep to an encoder
        # node when there's pipeline budget; otherwise fall through to
        # the regular least-recent-step tie-breaker.
        async_node, async_walk = self._maybe_pick_async_encoder(
            node_name_to_requests
        )
        if async_node is not None:
            return async_node, async_walk

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

        # MSTAR_ENCODER_ASYNC depth bookkeeping. We count a credit each time
        # an encoder batch is *scheduled* (regardless of batch size, since the
        # depth bound is about pipeline lead, not memory footprint per batch),
        # and release a credit when the matching Thinker prefill walk runs.
        # The Sequential[encoder, Thinker] structure of ``prefill_audio`` /
        # ``prefill_vision`` guarantees the encoder fires exactly once per
        # walk on the Thinker side, so this counter stays bounded as long as
        # the Thinker actually consumes the buffered embeddings. If a request
        # is removed mid-flight (RemoveRequest before the Thinker step ran),
        # the credit is reclaimed by ``release_encoder_async_credit`` from
        # the worker's remove path so the counter cannot drift upward.
        if self.encoder_async_enabled:
            if best_node_name in _ENCODER_NODE_NAMES:
                self.encoder_async_in_flight += 1
            elif (
                best_node_name == "Thinker"
                and graph_walk in _ENCODER_CONSUMING_WALKS
                and self.encoder_async_in_flight > 0
            ):
                # One Thinker prefill_vision/prefill_audio step consumes one
                # buffered batch of encoder outputs. The encoder may have
                # produced N requests' worth of embeds in a single batched
                # call (MSTAR_BATCH_VISION_PREFILL), but the Thinker side
                # consumes them sequentially — one walk per request. The
                # accounting here is per encoder *batch*, so as long as a
                # single Thinker walk releases a credit we'll always have
                # capacity to keep the encoder one batch ahead.
                self.encoder_async_in_flight -= 1

        return ScheduledBatch(
            node_name=best_node_name,
            graph_walk=graph_walk,
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
        )

    def release_encoder_async_credit(self, count: int = 1) -> None:
        """Release ``count`` credits from the encoder-async in-flight counter.

        Called when a request is torn down before its buffered encoder
        output is consumed by the Thinker (e.g. RemoveRequest mid-flight,
        or a hard failure that drops the Thinker prefill step). Without
        this, the counter would drift up and eventually saturate the
        depth budget, silently disabling the async pipeline path.

        No-op when the flag is off or the counter is already at zero.
        """
        if not self.encoder_async_enabled:
            return
        self.encoder_async_in_flight = max(
            0, self.encoder_async_in_flight - count,
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
