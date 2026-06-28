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

# Thinker prefill walks eligible for resumable chunked prefill. Kept local to
# the scheduler so it does not import the (heavy) Qwen3-Omni model module just
# to read the env flag.
_THINKER_PREFILL_WALKS = frozenset(
    {"prefill_text", "prefill_audio", "prefill_vision"}
)


def _chunked_prefill_enabled() -> bool:
    """Mirror of ``qwen3_omni_model.chunked_prefill_enabled`` without importing
    the model module (avoids a worker->model import cycle). Default OFF."""
    raw = os.environ.get("MSTAR_CHUNKED_PREFILL")
    return raw is not None and raw.strip().lower() in ("1", "true", "yes", "on")


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

        # Resumable chunked prefill (MSTAR_CHUNKED_PREFILL): cap new prefill
        # tokens per step and re-enqueue the remainder. STUBBED -- see the
        # method docstring for exactly what remains. Dormant by default and
        # when the flag is OFF; only fires once the conductor marks a request
        # as needing chunking, which it does not yet.
        if _chunked_prefill_enabled() and graph_walk in _THINKER_PREFILL_WALKS:
            self._maybe_reenqueue_prefill_remainder(
                worker_graphs_manager, graph_walk, node_objects,
            )

        return ScheduledBatch(
            node_name=best_node_name,
            graph_walk=graph_walk,
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
        )

    def _maybe_reenqueue_prefill_remainder(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        graph_walk: str,
        node_objects: dict[str, GraphNode],
    ) -> None:
        """STUB: cap each request's prefill to the token threshold and put the
        remainder back on its queue for the next scheduler cycle.

        This is the one piece of resumable chunked prefill that is NOT yet
        implemented (it cannot be validated without a GPU). The model side is
        complete: ``ThinkerSubmodule._maybe_chunk_prefill`` slices a precomputed
        full-span prefill (embeds + 3D M-RoPE + talker masks) to the window
        ``[prefill_chunk_offset : prefill_chunk_offset + prefill_chunk_len]``
        read from ``fwd_info.step_metadata``, and the FlashInfer KV append is
        already resumable (``cache_manager._plan_attention_impl`` writes ``sl``
        new tokens at offset ``state.seq_len`` and grows pages to
        ``state.seq_len + sl``), so feeding chunks across steps appends KV
        correctly and the next chunk's queries attend causally over the KV the
        previous chunks wrote.

        TODO -- to finish, in priority order:
          1. Per-request computed-tokens counter. Track new prefill tokens
             admitted so far for each (request_id, prefill walk). The total
             span length is known to the conductor (text token count;
             audio/vision token count after the encoder node runs), not to the
             scheduler -- so the conductor must publish it (e.g. on
             ``CurrentForwardPassInfo.step_metadata['prefill_total_len']``).
          2. Cap + re-enqueue. When ``computed + threshold < total``, set
             ``step_metadata['prefill_chunk_offset'] = computed`` and
             ``['prefill_chunk_len'] = min(threshold, total - computed)`` for
             THIS step, advance the counter, and re-push the SAME prefill node
             onto the request's per-request queue (``queue.push_ready_node`` /
             the WorkerGraphsManager re-enqueue API) so it is ready again next
             cycle. Only the FINAL chunk sets ``is_last_prefill=True`` (so
             logits are sampled once -- see ThinkerSubmodule.forward ~L868 and
             the conductor ~L1098).
          3. Conductor coordination (qwen3_omni_model.py). The Thinker state
             machine (``_get_thinker_forward``) must NOT advance
             ``prefill_step`` until the current walk's span is fully consumed,
             and the Talker's ``num_thinker_prefill_steps`` must keep counting
             one streamed thinker_states chunk per WALK, not per token-chunk
             (or the Talker last-prefill detection drifts). Simplest: stream
             thinker_states only on the final token-chunk of each walk.
          4. Encoder-output persistence for audio/vision. The encoder node runs
             once and its output (audio_embeds / vision_embeds + deepstack) must
             persist across the walk's chunk steps; today it is consumed by the
             single Thinker step. prefill_vision additionally needs per-chunk
             deepstack_<i> slicing (ThinkerSubmodule._maybe_chunk_prefill raises
             NotImplementedError for vision until then).
          5. seen_token_mask: add_tokens must run once per token (currently per
             walk step on the full span); move it to slice on the chunk window.
          6. CUDA graphs: a fixed chunk size == one PREFILL_TOKEN_BUCKETS entry,
             so full chunks replay an existing capture; the ragged final chunk
             uses the smallest bucket >= its size, exactly like a short
             single-shot prefill today. No new captures required.

        Validation is GPU-only: assert 2-chunk prefill KV + first-token logits
        match single-shot (test/modular/test_qwen3_omni_chunked_prefill_parity.py).
        """
        needs_chunking = any(
            getattr(n, "requires_prefill_chunking", False)
            for n in node_objects.values()
        )
        if not needs_chunking:
            # No conductor-marked request needs chunking -> dormant. Short
            # prefills (<= threshold) and the flag-ON single-chunk path reach
            # here and proceed unchanged.
            return
        raise NotImplementedError(
            "Resumable chunked-prefill re-enqueue is not implemented; see "
            "MicroScheduler._maybe_reenqueue_prefill_remainder for the full "
            "TODO and DESIGN_chunked_prefill.md. Unset MSTAR_CHUNKED_PREFILL "
            "to use single-shot prefill."
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
