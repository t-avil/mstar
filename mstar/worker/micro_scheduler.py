import logging
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

    def __init__(
        self, engine_manager: EngineManager,
        sched_type=SchedulingType.ROUND_ROBIN,
        tp_rank_zero_nodes: set[str] | None = None,
        max_consec_tp_follower_batches: int = 1,
        tp_nodes: set[str] | None = None,
    ):
        self.engine_manager = engine_manager
        self.batch_number = 0
        self.sched_type = sched_type

        # tensor parallel
        self.tp_rank_zero_nodes = tp_rank_zero_nodes
        # Nodes with TP world_size > 1 (distinct from tp_rank_zero_nodes, which
        # also includes every non-TP node since those are rank 0). Used to keep
        # W5-P2 mixed-batch assembly off TP nodes: a mixed batch's per-request
        # walks are heterogeneous and TP fan-out (ScheduleTPNode / sharding
        # group lookup) is keyed by a single batch walk, so a "thinker_mixed"
        # TP batch has no sharding group. TP mixed is P3.
        self.tp_nodes = tp_nodes or set()
        self.tp_batches_pending_schedule = deque()
        self.num_consec_tp_follower_batches = 0
        self.max_consec_tp_follower_batches = max_consec_tp_follower_batches

        self.node_and_walk_to_last_batch_num = {}
        # W5-P2 mixed-batch: count of thinker_mixed batches assembled by this
        # scheduler. Surfaced in the per-assembly INFO log; a monotonic counter
        # gives runtime evidence the mixed path is firing without DEBUG.
        self.mixed_batches_assembled = 0
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


    # W5-P2 mixed-batch walk names + capacity. Kept as module-adjacent
    # constants (mirroring ThinkerSubmodule.MIXED_BATCH_*) so the scheduler
    # does not import the model submodule. If those change, change these.
    _MIXED_DECODE_WALK = "thinker_decode"
    _MIXED_CHUNK_WALK = "prefill_text"
    # W5-P3-lite: a VISION prefill chunk may also serve as the mixed step's
    # single chunk row when MSTAR_MIXED_BATCH_VISION is on. Same C caps; the
    # Thinker's mixed capture then carries deepstack statics + the MRoPE
    # side-channel (see ThinkerSubmodule.preprocess / get_cuda_graph_configs).
    _MIXED_VISION_CHUNK_WALK = "prefill_vision"
    _MIXED_MAX_DECODE = 31          # padded_bs 32 = up to 31 decode + 1 chunk row
    _MIXED_MAX_CHUNK_TOKENS = 512   # largest captured chunk bucket (C in {256,512})

    def _mixed_min_decode(self) -> int:
        """Occupancy floor for chain-folding (see has_mixed_opportunity).

        Default 0 (no floor — preserves the measured-positive P2 behavior,
        including small-batch s2t folds) UNLESS an EAGER (every-step) fold policy
        is on — MSTAR_MIXED_SINGLE_CHUNK or MSTAR_MIXED_BUDGET_TOKENS — where
        every admission depends on fold slots and a small decode side means
        folding throttles admission: default 24 there. The occupancy floor is
        the graveyard's second anti-lesson (single-chunk starved decode ~10%),
        so the eager budget policy inherits it.

        Overrides (precedence): MSTAR_MIXED_BUDGET_MIN_DECODE (the V2 knob) wins,
        else MSTAR_MIXED_MIN_DECODE (the general one), else the default above.
        Cached after first read; reset by _refresh_dynamic_flags on a dynflags
        flip so a runtime budget toggle re-derives the floor."""
        v = getattr(self, "_mixed_min_decode_cached", None)
        if v is None:
            import os
            from mstar.model.qwen3_omni.qwen3_omni_model import (
                mixed_single_chunk_enabled,
                mixed_budget_tokens,
            )
            eager = mixed_single_chunk_enabled() or mixed_budget_tokens() > 0
            default = 24 if eager else 0
            raw = os.environ.get("MSTAR_MIXED_BUDGET_MIN_DECODE")
            if raw is None:
                raw = os.environ.get("MSTAR_MIXED_MIN_DECODE")
            try:
                v = int(raw) if raw is not None else default
            except ValueError:
                v = default
            self._mixed_min_decode_cached = v
        return v

    def _mixed_chunk_walks(self) -> set[str]:
        """Walks that may serve as a mixed step's single chunk row. prefill_text
        always; prefill_vision only when MSTAR_MIXED_BATCH_VISION is on (the
        vision chunk carries deepstack + MRoPE, replayable only by the
        vision-capable mixed capture — see _try_assemble_mixed)."""
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            mixed_batch_vision_enabled,
        )
        walks = {self._MIXED_CHUNK_WALK}
        if mixed_batch_vision_enabled():
            walks.add(self._MIXED_VISION_CHUNK_WALK)
        return walks

    def _chunk_entry_passes_gates(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        node_name: str,
        node_partition,
        entry: "ReadyNodeEntry",
    ) -> bool:
        """Per-request gates a chunk row must pass to join a mixed batch.

        Shared by the assembler (_try_assemble_mixed) and the read-only peek
        (has_mixed_opportunity) so the two never diverge on what counts as a
        mixable chunk. Gates (see _try_assemble_mixed docstring): chunk metadata
        present (P1 chunked, not a full unchunked prefill), C within the largest
        captured bucket, and repetition_penalty == 1.0 (a discarded chunk sample
        must not perturb penalty state)."""
        fwd_info = worker_graphs_manager.get_fwd_info(
            entry.request_id, node_partition,
        )
        clen = fwd_info.step_metadata.get("prefill_chunk_len")
        if clen is None:
            return False  # unchunked full prefill — don't mix (bucket blow)
        if int(clen) > self._MIXED_MAX_CHUNK_TOKENS:
            return False
        sc = fwd_info.sampling_config.get(node_name)
        if sc is not None and getattr(sc, "repetition_penalty", 1.0) != 1.0:
            return False  # penalty state corruption on discarded chunk sample
        return True

    def _chunk_over_budget(
        self, clen, n_decode: int | None, budget_tokens: int,
    ) -> bool:
        """V2 budget gate (MSTAR_MIXED_BUDGET_TOKENS): a chunk is over budget
        when the policy is on (budget > 0) and folding it would push the mixed
        step past ``budget`` total tokens (``n_decode`` 1-token decode rows + the
        ``C``-token chunk). budget <= 0 (off) or an unknown decode size / chunk
        length never binds. Shared by the peek (has_mixed_opportunity) and the
        pop (pop_mixed_chunk_for_spec) so the two agree on which chunk folds."""
        if budget_tokens <= 0 or n_decode is None or clen is None:
            return False
        return n_decode + int(clen) > budget_tokens

    def has_mixed_opportunity(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        decode_target: tuple[str, str],
        n_decode: int | None = None,
        budget_tokens: int = 0,
    ) -> bool:
        """Read-only peek: would a mixed batch assemble RIGHT NOW if the decode
        group named by ``decode_target`` were back in the ready queue?

        The worker calls this while a decode chain is in flight — the decode
        rids are ``_speculatively_scheduled`` and thus absent from the ready
        scan (base.py register_ingested_input gate) — to decide whether to break
        the chain into the NON-speculative path so ``get_next_batch`` can
        assemble decode + chunk into a ``thinker_mixed`` batch there. Because the
        decode rids are absent, this peek only needs to confirm a mixable CHUNK
        is ready on the decode's node; the decode side is guaranteed to re-enter
        the queue once the in-flight step completes and un-flags.

        Mirrors the gates in ``_try_assemble_mixed`` without popping or mutating
        queue state. Returns False when the flag is off (so the default
        yield-away path is byte-identical when mixed batching is disabled).

        ``n_decode``: size of the in-flight decode chain, when the caller knows
        it. Folding admits at most ONE chunk per step, so during ramp-up (small
        decode side, many requests still prefilling) folding THROTTLES admission
        and starves decode occupancy — measured 6.18 -> 3.48 req/s at i2t B32
        when every short span became foldable (MSTAR_MIXED_SINGLE_CHUNK).
        Standalone prefill fills the batch faster there. Gate: fold only when
        n_decode >= MSTAR_MIXED_MIN_DECODE (default 24); None skips the gate
        (non-spec assembler paths size the decode side themselves).

        ``budget_tokens``: V2 per-step token budget (MSTAR_MIXED_BUDGET_TOKENS).
        When > 0, a chunk only counts as an opportunity if n_decode + C fits the
        budget; the scan keeps looking for a smaller chunk otherwise. 0 = off
        (no cap), so P2 / yield-boundary behavior is byte-identical.
        """
        from mstar.model.qwen3_omni.qwen3_omni_model import mixed_batch_enabled
        if not mixed_batch_enabled():
            return False
        if n_decode is not None and n_decode < self._mixed_min_decode():
            return False

        decode_node_name, decode_walk = decode_target
        if decode_walk != self._MIXED_DECODE_WALK:
            return False
        if decode_node_name in self.tp_nodes:
            return False  # TP mixed batches are P3

        chunk_walks = self._mixed_chunk_walks()
        node_partition = worker_graphs_manager.get_partition_for_node(
            decode_node_name
        )
        now = time.monotonic()
        for _wg_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            for request_id, node_names in ready_map.items():
                if request_id not in worker_graphs_manager.per_request_info:
                    continue
                if request_id in self.pending_removes:
                    continue
                if request_id in self.held_until and self.held_until[request_id] > now:
                    continue
                if decode_node_name not in node_names:
                    continue
                if decode_node_name not in self.tp_rank_zero_nodes:
                    continue
                walk = worker_graphs_manager.get_graph_walk(
                    request_id, node_partition,
                )
                if walk not in chunk_walks:
                    continue
                fwd_info = worker_graphs_manager.get_fwd_info(
                    request_id, node_partition,
                )
                engine = self.engine_manager.get_engine(decode_node_name)
                if not engine.check_ready(decode_node_name, request_id, fwd_info):
                    continue
                entry = ReadyNodeEntry(request_id, _wg_id, walk)
                if not self._chunk_entry_passes_gates(
                    worker_graphs_manager, decode_node_name, node_partition, entry,
                ):
                    continue
                clen = fwd_info.step_metadata.get("prefill_chunk_len")
                if self._chunk_over_budget(clen, n_decode, budget_tokens):
                    continue  # over budget with this decode side; keep scanning
                return True
        return False

    def _try_assemble_mixed(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        node_name_to_requests: dict[str, list["ReadyNodeEntry"]],
        max_batch_size: int | None,
    ) -> "ScheduledBatch | None":
        """Assemble a mixed thinker_decode + prefill_text-chunk batch (W5-P2).

        Returns a ScheduledBatch with graph_walk="thinker_mixed" covering all
        ready decode rows on a node plus ONE ready prefill_text chunk row, or
        None when the flag is off / no node has both / a gate fails (in which
        case ``get_next_batch`` falls through to the normal single-walk path).

        Per-request CurrentForwardPassInfo.graph_walk is NOT touched here — each
        popped node keeps its own walk (thinker_decode / prefill_text), which is
        what the submodule's prepare_inputs / postprocess dispatch on. Only the
        batch-level graph_walk is "thinker_mixed" (drives config + runner).

        Gates (any failing → None, fall back to P1 alternation):
          * MSTAR_MIXED_BATCH on.
          * A node with BOTH a decode group and >=1 prefill_text chunk row.
          * The chunk row carries P1 chunk metadata (prefill_chunk_len set): a
            full unchunked prefill is not mixed (it would blow the token bucket).
          * Chunk C <= _MIXED_MAX_CHUNK_TOKENS so (n + C) fits a captured bucket.
          * Chunk request repetition_penalty == 1.0: a non-last chunk row's
            sampled token is discarded (postprocess drops it), but Sampler.sample
            adds every sampled token to the seen-token mask + advances the RNG
            when any rep-penalty is active, which would corrupt the chunk
            request's penalty state. Decode rows are unaffected. (design D gate)
        """
        from mstar.model.qwen3_omni.qwen3_omni_model import mixed_batch_enabled
        if not mixed_batch_enabled():
            return None

        # W5-P3-lite: allow a prefill_vision chunk row alongside prefill_text
        # when the vision flag is on. A vision chunk carries deepstack + the
        # MRoPE side-channel, which only the vision-capable mixed capture can
        # replay; with the flag off the chunk row stays prefill_text (P2).
        chunk_walks = self._mixed_chunk_walks()

        for node_name, entries in node_name_to_requests.items():
            if node_name in self.tp_nodes:
                continue  # TP mixed batches are P3 (see __init__ note)
            decode_entries = [
                e for e in entries if e.graph_walk == self._MIXED_DECODE_WALK
            ]
            chunk_entries = [
                e for e in entries if e.graph_walk in chunk_walks
            ]
            if not decode_entries or not chunk_entries:
                continue
            # Occupancy floor (see _mixed_min_decode): a small decode side
            # makes the mixed step poor value AND throttles admission (one
            # chunk per step). Let prefills run standalone instead. Floor is
            # 0 unless MSTAR_MIXED_SINGLE_CHUNK, so P2 behavior is unchanged.
            if len(decode_entries) < self._mixed_min_decode():
                continue

            node_partition = worker_graphs_manager.get_partition_for_node(node_name)

            # Pick the first chunk entry that passes the per-request gates.
            chunk_entry = None
            for e in chunk_entries:
                if self._chunk_entry_passes_gates(
                    worker_graphs_manager, node_name, node_partition, e,
                ):
                    chunk_entry = e
                    break
            if chunk_entry is None:
                continue

            # Cap decode rows so total rows (decode + 1 chunk) fit the padded_bs
            # bucket, honoring any caller max_batch_size too.
            cap = self._MIXED_MAX_DECODE
            if max_batch_size is not None:
                cap = min(cap, max_batch_size - 1)
            if cap < 1:
                continue
            decode_entries = decode_entries[:cap]

            batch = self._pop_mixed_batch(
                worker_graphs_manager, node_name, decode_entries, chunk_entry,
            )
            if batch is not None:
                return batch
        return None

    def _pop_mixed_batch(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        node_name: str,
        decode_entries: list["ReadyNodeEntry"],
        chunk_entry: "ReadyNodeEntry",
    ) -> "ScheduledBatch | None":
        """Pop the selected decode + chunk nodes and build the mixed batch.

        Mirrors the pop loop in get_next_batch. If nothing pops (races with a
        removal), returns None so the caller falls back to the normal path.
        """
        node_partition = worker_graphs_manager.get_partition_for_node(node_name)
        chunk_fwd_info = worker_graphs_manager.get_fwd_info(
            chunk_entry.request_id, node_partition,
        )
        chunk_len = chunk_fwd_info.step_metadata.get("prefill_chunk_len")

        node_objects = {}
        request_to_worker_graph = {}
        for entry in [*decode_entries, chunk_entry]:
            queue = worker_graphs_manager.queues[entry.worker_graph_id]
            popped = queue.pop_ready_nodes(entry.request_id, [node_name])
            if popped:
                assert len(popped) == 1
                node_objects[entry.request_id] = popped[0]
                request_to_worker_graph[entry.request_id] = entry.worker_graph_id

        # Require at least one decode row AND the chunk row to have popped;
        # a lone chunk is just a normal prefill and should go the normal path.
        if chunk_entry.request_id not in node_objects or len(node_objects) < 2:
            return None

        self.batch_number += 1
        self.node_and_walk_to_last_batch_num[(node_name, "thinker_mixed")] = \
            self.batch_number
        n_decode = len(node_objects) - 1
        self.mixed_batches_assembled += 1
        # INFO (not DEBUG): one line per assembly so runtime evidence that mixed
        # batching is actually firing doesn't require a DEBUG flood. n_decode/C
        # are the batch shape; total = the assembled count so far this worker.
        logger.info(
            "mixed batch: n_decode=%d C=%s node=%s chunk_rid=%s total=%d",
            n_decode, chunk_len, node_name,
            chunk_entry.request_id, self.mixed_batches_assembled,
        )
        return ScheduledBatch(
            node_name=node_name,
            graph_walk="thinker_mixed",
            node_objects=node_objects,
            request_to_worker_graph=request_to_worker_graph,
        )

    def pop_mixed_chunk_for_spec(
        self,
        worker_graphs_manager: WorkerGraphsManager,
        decode_target: tuple[str, str],
        n_decode: int | None = None,
        budget_tokens: int = 0,
    ) -> "tuple[GraphNode, str, str, int | None] | None":
        """Pop ONLY the mixable chunk node for a mid-chain mixed speculation
        (MSTAR_MIXED_SPEC).

        Unlike ``_pop_mixed_batch``, this does NOT touch the decode rows: during
        a live decode spec chain the decode rids are ``_speculatively_scheduled``
        and absent from the ready queue — they continue via the worker's
        speculation machinery (registry nodes made ready by
        ``ingest_for_speculation``), NOT via the ready queue. The worker builds
        the decode continuation itself and calls this to obtain the single chunk
        row to fold in.

        Scans exactly like ``has_mixed_opportunity`` (same gates, via
        ``_chunk_entry_passes_gates``, and the same ``n_decode`` /
        ``budget_tokens`` V2 budget filter) and pops the FIRST passing chunk node
        on ``decode_target``'s node. Returns
        ``(chunk_node, request_id, worker_graph_id, prefill_chunk_len)`` or None
        when the flag is off / no mixable chunk is ready / the pop races a
        removal. The caller injects the returned node into the speculative
        ScheduledBatch under ``graph_walk="thinker_mixed"``.

        Guarded by ``mixed_batch_spec_enabled`` so the whole path is unreachable
        (and thus byte-identical) unless MSTAR_MIXED_SPEC is on.
        """
        from mstar.model.qwen3_omni.qwen3_omni_model import (
            mixed_batch_spec_enabled,
        )
        if not mixed_batch_spec_enabled():
            return None

        decode_node_name, decode_walk = decode_target
        if decode_walk != self._MIXED_DECODE_WALK:
            return None
        if decode_node_name in self.tp_nodes:
            return None  # TP mixed batches are P3

        chunk_walks = self._mixed_chunk_walks()
        node_partition = worker_graphs_manager.get_partition_for_node(
            decode_node_name
        )
        now = time.monotonic()
        for wg_id, queue in worker_graphs_manager.queues.items():
            ready_map = queue.get_ready_node_names()
            # Snapshot the rids so popping mid-iteration can't mutate the map we
            # are scanning.
            for request_id, node_names in list(ready_map.items()):
                if request_id not in worker_graphs_manager.per_request_info:
                    continue
                if request_id in self.pending_removes:
                    continue
                if request_id in self.held_until and self.held_until[request_id] > now:
                    continue
                if decode_node_name not in node_names:
                    continue
                if decode_node_name not in self.tp_rank_zero_nodes:
                    continue
                walk = worker_graphs_manager.get_graph_walk(
                    request_id, node_partition,
                )
                if walk not in chunk_walks:
                    continue
                fwd_info = worker_graphs_manager.get_fwd_info(
                    request_id, node_partition,
                )
                engine = self.engine_manager.get_engine(decode_node_name)
                if not engine.check_ready(decode_node_name, request_id, fwd_info):
                    continue
                entry = ReadyNodeEntry(request_id, wg_id, walk)
                if not self._chunk_entry_passes_gates(
                    worker_graphs_manager, decode_node_name, node_partition, entry,
                ):
                    continue
                chunk_len = fwd_info.step_metadata.get("prefill_chunk_len")
                # V2 budget gate: mirror has_mixed_opportunity so pop selects the
                # SAME first budget-fitting chunk the peek approved (else a bigger
                # chunk could pop and blow the budget the peek respected).
                if self._chunk_over_budget(chunk_len, n_decode, budget_tokens):
                    continue
                popped = queue.pop_ready_nodes(request_id, [decode_node_name])
                if not popped:
                    continue  # raced a removal; keep scanning
                assert len(popped) == 1
                return popped[0], request_id, wg_id, chunk_len
        return None

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

        # W5-P2 mixed prefill+decode batch (MSTAR_MIXED_BATCH). Before the
        # normal one-walk-per-batch selection, try to assemble a mixed batch
        # (N thinker_decode rows + 1 prefill_text chunk row) so the captured
        # thinker_mixed graph runs both kinds in one forward. Returns None
        # (flag off, no opportunity, or gates fail) → fall through to the
        # normal path, byte-identical to today.
        mixed = self._try_assemble_mixed(
            worker_graphs_manager, node_name_to_requests, max_batch_size,
        )
        if mixed is not None:
            return mixed

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
