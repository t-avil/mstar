import logging
import queue
from dataclasses import dataclass, field

import torch

from mminf.engine.base import BaseEngine, EngineType, NodeBatch, NodeOutput
from mminf.utils.profiler import range_pop, range_push

logger = logging.getLogger(__name__)

@dataclass
class KVCacheConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_seq_len: int
    max_num_pages: int = 2048
    page_size: int = 128
    num_qo_heads: int | None = None  # Optional, defaults to num_kv_heads

    def __post_init__(self):
        if self.num_qo_heads is None:
            self.num_qo_heads = self.num_kv_heads


@dataclass
class KVRequestState:
    """Per-request KV cache state for the AR engine."""
    page_indices: list[int] = field(default_factory=list)
    seq_len: int = 0
    position_id_start: int = 0
    is_paused: bool = False


class PageAllocator:
    """Simple page allocator using a FIFO queue of free page indices."""

    def __init__(self, max_num_pages: int):
        self.max_num_pages = max_num_pages
        self.free_pages: queue.Queue[int] = queue.Queue()
        for i in range(max_num_pages):
            self.free_pages.put(i)

    def allocate(self, n: int) -> list[int]:
        if self.free_pages.qsize() < n:
            raise RuntimeError(
                f"Not enough free pages: requested {n}, "
                f"available {self.free_pages.qsize()}"
            )
        return [self.free_pages.get() for _ in range(n)]

    def free(self, pages: list[int]) -> None:
        for page in pages:
            self.free_pages.put(page)

    @property
    def num_free(self) -> int:
        return self.free_pages.qsize()


class CacheHandle:
    """Per-request cache handle. Created by AR engine, passed to submodule.forward().

    Provides an interface between the engine's cache infrastructure (FlashInfer,
    page tables, KV tensor) and the submodule's cache semantics (which caches
    to read/write, when to snapshot, how to combine multi-cache outputs).

    The active_label determines which KV cache subsequent run_attention calls
    operate on. For single-cache models this is always "main"; for CFG models
    like BAGEL it switches between "main", "cfg_text", "cfg_img".
    """

    def __init__(
        self,
        request_id: str,
        kv_cache: torch.Tensor | None,
        page_allocator: PageAllocator | None,
        request_states: dict[tuple[str, str], KVRequestState],
        workspace_buffer: torch.Tensor | None,
        kv_cache_config: KVCacheConfig,
        device,
    ):
        self.request_id = request_id
        self.kv_cache = kv_cache
        self.page_allocator = page_allocator
        self.request_states = request_states
        self.workspace_buffer = workspace_buffer
        self.kv_cache_config = kv_cache_config
        self.max_seq_len = kv_cache_config.max_seq_len
        self.active_label = "main"

        # for RoPE, may be moved!
        self.base_pos_ids = torch.arange(
            self.max_seq_len, dtype=torch.long, device=device
        )
        self.device = device

    def _get_state(self, label: str | None = None) -> KVRequestState:
        label = label or self.active_label
        key = (self.request_id, label)
        if key not in self.request_states:
            self.request_states[key] = KVRequestState()
        return self.request_states[key]

    def set_active_label(self, label: str) -> None:
        """Switch which KV cache subsequent run_attention calls use."""
        self.active_label = label


    def run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        is_causal: bool = True,
        write_cache: bool = True,
    ) -> torch.Tensor:
        """Run paged attention for the active cache label at this layer.

        Reads existing K,V from the active label's cache pages, concatenates
        with the new k,v, computes attention, and optionally writes new k,v
        to the cache.

        write_cache=False: use prefix cache for attention but don't persist
        new k,v. Used during flow matching where latent tokens are re-processed
        fresh each step against a frozen prefix.

        Args:
            q: [seq_len, num_q_heads, head_dim]
            k: [seq_len, num_kv_heads, head_dim]
            v: [seq_len, num_kv_heads, head_dim]
            layer_idx: transformer layer index
            is_causal: whether to apply causal masking
            write_cache: whether to persist new k,v to cache pages

        Returns:
            Attention output tensor [seq_len, num_q_heads, head_dim]
        """
        # if self.kv_cache is None:
        #     # Dummy mode: return zeros shaped like q
        #     return torch.zeros_like(q)

        assert self.kv_cache is not None

        state = self._get_state()
        cfg = self.kv_cache_config
        page_size = cfg.page_size
        num_kv_heads = cfg.num_kv_heads
        head_dim = cfg.head_dim
        num_qo_heads = cfg.num_qo_heads
        seq_len = q.shape[0]

        # Allocate pages if needed for the new tokens
        total_len = state.seq_len + seq_len
        num_pages_needed = (total_len + page_size - 1) // page_size
        num_new_pages = num_pages_needed - len(state.page_indices)
        if num_new_pages > 0:
            new_pages = self.page_allocator.allocate(num_new_pages)
            state.page_indices.extend(new_pages)

        # Write K,V into cache pages at this layer
        for i in range(seq_len):
            pos = state.seq_len + i
            page_idx = state.page_indices[pos // page_size]
            offset = pos % page_size
            self.kv_cache[layer_idx, page_idx, 0, offset] = k[i]
            self.kv_cache[layer_idx, page_idx, 1, offset] = v[i]

        # logger.warning(f"write_cache is True. num_pages_needed={num_pages_needed}, num_new_pages={num_new_pages}")

        # Build FlashInfer single-request prefill args
        device = q.device
        kv_indptr = torch.tensor(
            [0, len(state.page_indices)], dtype=torch.int32, device=device
        )
        kv_indices = torch.tensor(
            state.page_indices, dtype=torch.int32, device=device
        )

        # logger.warning(f"kv_indptr={kv_indptr}, kv_indices={kv_indices}")
        # logger.warning(f"q.shape = {q.shape}, k.shape = {k.shape}, v.shape = {v.shape}")

        total_len = state.seq_len + seq_len

        last_page_len = total_len % page_size or page_size
        kv_last_page_len = torch.tensor(
            [last_page_len], dtype=torch.int32, device=device
        )

        try:
            import flashinfer
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )

            qo_indptr = torch.tensor(
                [0, seq_len], dtype=torch.int32, device=device
            )
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=kv_indices,
                paged_kv_last_page_len=kv_last_page_len,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim,
                page_size=page_size,
                causal=is_causal,
                q_data_type=q.dtype,
            )

            # q shape for FlashInfer: [total_tokens, num_heads, head_dim]
            output = wrapper.run(q, self.kv_cache[layer_idx])
        except ImportError as e:
            # No FlashInfer: naive attention fallback (for testing)
            logger.error("Could not run flashinfer. Outputting zeros from run_attention()")
            raise e

        return output


    def apply_rope_default(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rotary_dim: int | None = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 10000.0,
    ):
        offset = self._get_state().position_id_start
        pos_ids = self.base_pos_ids[:q.shape[0]] + offset

        import flashinfer
        return flashinfer.rope.apply_rope_pos_ids(
            q, k, pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )

    def apply_rope_custom_pos_ids(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        pos_ids: torch.Tensor,
        rotary_dim: int | None = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 10000.0,
    ):
        import flashinfer
        return flashinfer.rope.apply_rope_pos_ids(
            q, k, pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )

    def snapshot(self, from_label: str, to_label: str) -> None:
        """Deepcopy KV cache state (all layers, all pages) from one label to another.

        Allocates new pages for the target label and copies the KV data.
        """
        from_state = self._get_state(from_label)
        to_key = (self.request_id, to_label)

        if self.kv_cache is not None and self.page_allocator is not None:
            # Free old pages if target already exists (prevent page leak)
            if to_key in self.request_states:
                old_state = self.request_states[to_key]
                if old_state.page_indices:
                    self.page_allocator.free(old_state.page_indices)

            # Allocate fresh pages for the target
            num_pages = len(from_state.page_indices)
            new_pages = self.page_allocator.allocate(num_pages) if num_pages > 0 else []

            # Copy KV data page by page across all layers
            for src_page, dst_page in zip(from_state.page_indices, new_pages, strict=False):
                self.kv_cache[:, dst_page] = self.kv_cache[:, src_page]

            self.request_states[to_key] = KVRequestState(
                page_indices=new_pages,
                seq_len=from_state.seq_len,
                position_id_start=from_state.position_id_start,
            )
        else:
            # Dummy mode: just copy the state metadata
            self.request_states[to_key] = KVRequestState(
                page_indices=list(from_state.page_indices),
                seq_len=from_state.seq_len,
                position_id_start=from_state.position_id_start,
            )

    def advance_seq_len(self, n: int, pos_id_n: int | None=None) -> None:
        """Advance the active label's seq_len by n tokens.

        Must be called once after a full forward pass (all layers), not
        per-layer. This keeps seq_len consistent across layers for KV
        cache writes, RoPE offsets, and page allocation.
        """
        self._get_state().seq_len += n
        if pos_id_n is None:
            self._get_state().position_id_start += n
        else:
            self._get_state().position_id_start += pos_id_n

    def save_seq_position(self) -> int:
        """Return current seq_len for the active label (for flow matching rewind)."""
        return self._get_state().seq_len

    def restore_seq_position(self, pos: int) -> None:
        """Rewind the active label's seq_len (discard latent token KV entries).

        Does not deallocate pages — they will be overwritten on next write.
        """
        self._get_state().seq_len = pos


class BatchedCacheManager:
    """Manages batched FlashInfer attention for multiple requests simultaneously.

    Replaces per-request CacheHandle for decode and simple prefill batches where
    all requests use the same graph_walk. Constructs batch-level FlashInfer index
    tensors (qo_indptr, paged_kv_indptr, paged_kv_indices) and issues a single
    FlashInfer call per layer instead of N separate calls.

    Keep existing CacheHandle for backward compatibility — complex paths like
    image_gen (3-pass CFG with label switching) continue using per-request
    CacheHandle.
    """

    def __init__(
        self,
        request_ids: list[str],
        active_labels_per_request: dict[str, str],
        kv_cache: torch.Tensor,
        page_allocator: PageAllocator,
        request_states: dict[tuple[str, str], KVRequestState],
        workspace_buffer: torch.Tensor,
        kv_cache_config: KVCacheConfig,
        device,
    ):
        self.request_ids = request_ids
        self.active_labels = active_labels_per_request  # {req_id: label}
        self.kv_cache = kv_cache
        self.page_allocator = page_allocator
        self.request_states = request_states
        self.workspace_buffer = workspace_buffer
        self.kv_cache_config = kv_cache_config
        self.device = device

        self.wrapper = None
        self.page_indices: torch.Tensor | None = None
        self.page_offsets: torch.Tensor | None = None
        self.pos_ids: torch.Tensor | None = None

        self.base_pos_ids = torch.arange(
            kv_cache_config.max_seq_len, dtype=torch.long, device=device
        )

    def _get_state(self, request_id: str, label: str | None = None) -> KVRequestState:
        label = label or self.active_labels.get(request_id, "main")
        key = (request_id, label)
        if key not in self.request_states:
            self.request_states[key] = KVRequestState()
        return self.request_states[key]

    def set_active_labels(self, labels: dict[str, str]) -> None:
        """Switch active cache labels for all requests at once."""
        self.active_labels = labels

    def set_active_label(self, label: str) -> None:
        """Switch all requests to the same cache label."""
        self.active_labels = {rid: label for rid in self.request_ids}

    def plan_attention(
        self,
        seq_lens: list[int] | None = None,
        dtype=torch.bfloat16,
        is_causal=True
    ):
        assert self.kv_cache is not None

        cfg = self.kv_cache_config
        page_size = cfg.page_size
        num_kv_heads = cfg.num_kv_heads
        head_dim = cfg.head_dim
        num_qo_heads = cfg.num_qo_heads
        device = self.device

        # 1. For each request: allocate pages if needed, write K,V to cache
        qo_indptr_list = [0]
        kv_indptr_list = [0]
        all_page_indices = []
        kv_last_page_lens = []

        page_indices_all = []
        page_offsets_all = []
        for i, rid in enumerate(self.request_ids):
            state = self._get_state(rid)
            sl = seq_lens[i]

            # Allocate pages if needed
            total_len = state.seq_len + sl
            num_pages_needed = (total_len + page_size - 1) // page_size
            num_new_pages = num_pages_needed - len(state.page_indices)
            if num_new_pages > 0:
                new_pages = self.page_allocator.allocate(num_new_pages)
                state.page_indices.extend(new_pages)
            
             # Compute positions for this request
            pos = torch.arange(state.seq_len, state.seq_len + sl, device=self.device)

            page_idx = torch.tensor(state.page_indices, device=self.device)[pos // page_size]
            page_offset = pos % page_size

            page_indices_all.append(page_idx)
            page_offsets_all.append(page_offset)

            # Build indptr entries
            qo_indptr_list.append(qo_indptr_list[-1] + sl)
            all_page_indices.extend(state.page_indices)
            kv_indptr_list.append(kv_indptr_list[-1] + len(state.page_indices))

            last_page_len = total_len % page_size or page_size
            kv_last_page_lens.append(last_page_len)
        
        self.page_indices = torch.cat(page_indices_all)
        self.page_offsets = torch.cat(page_offsets_all)
        self.token_offsets = torch.arange(self.page_indices.numel(), device=self.device)

        # 2. Build batched FlashInfer index tensors
        qo_indptr = torch.tensor(qo_indptr_list, dtype=torch.int32, device=device)
        paged_kv_indptr = torch.tensor(kv_indptr_list, dtype=torch.int32, device=device)
        paged_kv_indices = torch.tensor(all_page_indices, dtype=torch.int32, device=device)
        paged_kv_last_page_len = torch.tensor(kv_last_page_lens, dtype=torch.int32, device=device)

        # 3. Plan + run BatchPrefillWithPagedKVCacheWrapper
        import flashinfer
        self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer, "NHD"
        )
        self.wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=num_qo_heads,
            num_kv_heads=num_kv_heads,
            head_dim_qk=head_dim,
            page_size=page_size,
            causal=is_causal,
            q_data_type=dtype,
        )
    
    def plan_rope(
        self,
        seq_lens: list[int],
        pos_ids: torch.Tensor | None = None
    ):
        if pos_ids is not None:
            self.pos_ids = pos_ids
            return
        self.pos_ids = torch.cat([
            torch.arange(sl, device=self.device, dtype=torch.long)
            + self._get_state(rid).position_id_start
            for rid, sl in zip(self.request_ids, seq_lens)
        ])

    def run_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Single FlashInfer call for all requests in the batch.

        When seq_lens is provided (batched path), q/k/v are concatenated tensors
        with shape [sum(seq_lens), heads, head_dim]. When seq_lens is None,
        falls back to treating q as a single request (backward compat).

        Args:
            q: [sum(seq_lens), num_q_heads, head_dim] - concatenated queries
            k: [sum(seq_lens), num_kv_heads, head_dim]
            v: [sum(seq_lens), num_kv_heads, head_dim]
            layer_idx: transformer layer index
            seq_lens: number of new tokens per request
        Returns:
            output: [sum(seq_lens), num_q_heads, head_dim]
        """
        assert self.kv_cache is not None, self.wrapper is not None
        # Two writes
        self.kv_cache[layer_idx, self.page_indices, 0, self.page_offsets] = k[self.token_offsets]
        self.kv_cache[layer_idx, self.page_indices, 1, self.page_offsets] = v[self.token_offsets]

        output = self.wrapper.run(q, self.kv_cache[layer_idx])
        return output

    def apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        rotary_dim: int | None = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 10000.0,
    ):
        """Apply RoPE to concatenated q, k with per-request position offsets."""
        assert self.pos_ids is not None

        import flashinfer
        return flashinfer.rope.apply_rope_pos_ids(
            q, k, self.pos_ids,
            rotary_dim=rotary_dim,
            interleave=interleave,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
        )

    def advance_seq_len(self, n: int, pos_id_n: int | None = None) -> None:
        """Advance seq_len for all requests by n tokens.

        Used when all requests in the batch process the same number of tokens
        (e.g., decode where each request generates 1 token).
        """
        for rid in self.request_ids:
            state = self._get_state(rid)
            state.seq_len += n
            if pos_id_n is None:
                state.position_id_start += n
            else:
                state.position_id_start += pos_id_n

    def advance_seq_lens(self, n_per_request: list[int], pos_id_ns: list[int] | None = None) -> None:
        """Advance seq_len for each request by different amounts."""
        for i, rid in enumerate(self.request_ids):
            state = self._get_state(rid)
            state.seq_len += n_per_request[i]
            if pos_id_ns is None:
                state.position_id_start += n_per_request[i]
            else:
                state.position_id_start += pos_id_ns[i]

    def snapshot_all(self, from_label: str, to_label: str) -> None:
        """Snapshot KV cache for all requests in batch."""
        for rid in self.request_ids:
            from_state = self._get_state(rid, from_label)
            to_key = (rid, to_label)

            # Free old pages if target already exists
            if to_key in self.request_states:
                old_state = self.request_states[to_key]
                if old_state.page_indices:
                    self.page_allocator.free(old_state.page_indices)

            num_pages = len(from_state.page_indices)
            new_pages = self.page_allocator.allocate(num_pages) if num_pages > 0 else []

            for src_page, dst_page in zip(from_state.page_indices, new_pages, strict=False):
                self.kv_cache[:, dst_page] = self.kv_cache[:, src_page]

            self.request_states[to_key] = KVRequestState(
                page_indices=new_pages,
                seq_len=from_state.seq_len,
                position_id_start=from_state.position_id_start,
            )


class AREngine(BaseEngine):
    """
    Autoregressive engine with paged KV cache.
    Uses FlashInfer for prefill/decode when available.
    Supports pause/resume for interleaved loops (LLM <-> flow).

    The engine provides cache infrastructure (FlashInfer, page tables, KV tensor)
    via CacheHandle objects. Submodules decide which caches to read/write, when
    to snapshot, and how to combine multi-cache outputs (e.g., CFG formula).
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig | dict,
        enable_nvtx: bool = False,
    ):
        super().__init__(enable_nvtx=enable_nvtx)
        if isinstance(kv_cache_config, dict):
            kv_cache_config = KVCacheConfig(**kv_cache_config)
        self.submodules: dict[str, torch.nn.Module] = {}
        self.kv_cache_config = kv_cache_config
        self.device = None
        self.kv_cache = None  # [num_layers, max_pages, 2, page_size, num_kv_heads, head_dim]
        self.page_allocator: PageAllocator | None = None
        # Keyed by (request_id, cache_label) to support multiple KV caches
        # per request (needed for BAGEL's classifier-free guidance which
        # maintains "main", "cfg_text", and "cfg_img" caches).
        self.request_states: dict[tuple[str, str], KVRequestState] = {}

        # FlashInfer wrappers (initialized in load_model if available)
        self.prefill_wrapper = None
        self.decode_wrapper = None
        self.workspace_buffer = None

        # CUDA graph runners (initialized in warmup())
        self.cuda_graph_runners: dict[str, "CudaGraphRunner"] = {}

    def engine_type(self) -> EngineType:
        return EngineType.AR

    def load_model(
        self,
        submodules: dict[str, torch.nn.Module],
        model_config: dict,
        device: torch.device,
    ) -> None:
        self.submodules = submodules
        self.device = device
        cfg = model_config.get(
            "kv_cache", self.kv_cache_config
        )
        if not cfg:
            return  # dummy mode without config
        if isinstance(cfg, dict):
            cfg = KVCacheConfig(**cfg)

        num_layers = cfg.num_layers
        max_num_pages = cfg.max_num_pages
        page_size = cfg.page_size
        num_kv_heads = cfg.num_kv_heads
        head_dim = cfg.head_dim

        self.kv_cache = torch.zeros(
            num_layers, max_num_pages, 2,
            page_size, num_kv_heads, head_dim,
            dtype=torch.bfloat16, device=device,
        )
        self.page_allocator = PageAllocator(max_num_pages)

        # 256MB workspace for FlashInfer
        self.workspace_buffer = torch.empty(
            256 * 1024 * 1024, dtype=torch.uint8, device=device
        )

        try:
            import flashinfer
            self.prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )
            self.decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )
        except ImportError:
            pass  # Dummy mode — no FlashInfer

    def _create_cache_handle(self, request_id: str) -> CacheHandle:
        """Create a CacheHandle for a single request."""
        return CacheHandle(
            request_id=request_id,
            kv_cache=self.kv_cache,
            page_allocator=self.page_allocator,
            request_states=self.request_states,
            workspace_buffer=self.workspace_buffer,
            kv_cache_config=self.kv_cache_config,
            device=self.device
        )

    def _compile_submodules(self) -> None:
        """Apply torch.compile to submodule forward paths.

        Uses mode="max-autotune-no-cudagraphs" (SGLang's approach) so compiled
        code gets baked into CUDA graphs when captured. Must be called BEFORE
        CUDA graph capture.
        """
        if not torch.cuda.is_available():
            return

        for node_name, submodule in self.submodules.items():
            if not hasattr(submodule, 'language_model'):
                continue
            try:
                lm = submodule.language_model
                lm.model.forward = torch.compile(
                    lm.model.forward,
                    mode="max-autotune-no-cudagraphs",
                    fullgraph=False,
                    dynamic=False,
                )
                logger.info("AREngine: torch.compile applied to %s language_model", node_name)
            except Exception:
                logger.warning("AREngine: torch.compile failed for %s, using eager mode",
                               node_name, exc_info=True)

    def warmup(self) -> None:
        """Compile submodules and capture CUDA graphs."""
        from mminf.engine.cuda_graph_runner import CudaGraphRunner

        if self.kv_cache is None or self.device is None:
            logger.info("AREngine: skipping warmup (no KV cache or device)")
            return

        # Step 1: torch.compile (before CUDA graph capture)
        self._compile_submodules()

        # Step 2: CUDA graph capture
        for node_name in self.submodules:
            runner = CudaGraphRunner(self, node_name, self.kv_cache_config)
            runner.warmup_and_capture()
            if runner.graphs:
                self.cuda_graph_runners[node_name] = runner
                logger.info("AREngine: CUDA graphs captured for %s (%d sizes)",
                            node_name, len(runner.graphs))

    def _can_batch(self, batch: NodeBatch) -> bool:
        """Only batch when all requests share a batchable graph_walk path.

        image_gen with 3-pass CFG is too complex to batch initially due to
        multi-label switching and snapshot operations within the forward pass.
        """
        if len(batch.request_ids) <= 1:
            return False
        if batch.graph_walk not in ("decode", "prefill_text"):
            return False
        # Ensure the submodule supports batched forward
        submodule = self.submodules.get(batch.node_name)
        if submodule is None or not hasattr(submodule, "forward_batched"):
            return False
        return True

    def _execute_batched(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Execute batch with BatchedCacheManager for true vectorized batching."""
        cache_manager = BatchedCacheManager(
            request_ids=batch.request_ids,
            active_labels_per_request={rid: "main" for rid in batch.request_ids},
            kv_cache=self.kv_cache,
            page_allocator=self.page_allocator,
            request_states=self.request_states,
            workspace_buffer=self.workspace_buffer,
            kv_cache_config=self.kv_cache_config,
            device=self.device,
        )

        # Preprocess all requests
        all_preprocessed = {}
        for rid in batch.request_ids:
            inputs = batch.per_request_input_tensors.get(rid, {})
            preprocessed = submodule.preprocess(batch.graph_walk, **inputs)
            all_preprocessed[rid] = preprocessed

        with torch.no_grad():
            batched_output = submodule.forward_batched(
                graph_walk=batch.graph_walk,
                cache_manager=cache_manager,
                per_request_inputs=all_preprocessed,
                per_request_metadata=batch.per_request_metadata,
            )

        return NodeOutput(per_request_output_tensors=batched_output)

    def _execute_sequential(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Original per-request execution with CacheHandle."""
        per_request_outputs = {}
        for rid in batch.request_ids:
            cache_handle = self._create_cache_handle(rid)
            inputs = batch.per_request_input_tensors.get(rid, {})
            metadata = batch.per_request_metadata.get(rid, {})

            preprocessed = submodule.preprocess(batch.graph_walk, **inputs)
            with torch.no_grad():
                output = submodule(
                    graph_walk=batch.graph_walk,
                    cache_handle=cache_handle,
                    **preprocessed,
                    **metadata,
                )
            per_request_outputs[rid] = output

        return NodeOutput(per_request_output_tensors=per_request_outputs)

    def _can_use_cuda_graph(self, batch: NodeBatch) -> bool:
        """Check if CUDA graph replay is available for this batch."""
        if batch.graph_walk != "decode":
            return False
        runner = self.cuda_graph_runners.get(batch.node_name)
        if runner is None:
            return False
        return runner.can_run(len(batch.request_ids))

    def _execute_with_cuda_graph(self, batch: NodeBatch, submodule) -> NodeOutput:
        """Execute using a captured CUDA graph."""
        runner = self.cuda_graph_runners[batch.node_name]

        cache_manager = BatchedCacheManager(
            request_ids=batch.request_ids,
            active_labels_per_request={rid: "main" for rid in batch.request_ids},
            kv_cache=self.kv_cache,
            page_allocator=self.page_allocator,
            request_states=self.request_states,
            workspace_buffer=self.workspace_buffer,
            kv_cache_config=self.kv_cache_config,
            device=self.device,
        )

        all_preprocessed = {}
        for rid in batch.request_ids:
            inputs = batch.per_request_input_tensors.get(rid, {})
            preprocessed = submodule.preprocess(batch.graph_walk, **inputs)
            all_preprocessed[rid] = preprocessed

        with torch.no_grad():
            batched_output = runner.run(
                batch_size=len(batch.request_ids),
                per_request_inputs=all_preprocessed,
                per_request_metadata=batch.per_request_metadata,
                cache_manager=cache_manager,
            )

        return NodeOutput(per_request_output_tensors=batched_output)

    def execute_batch(self, batch: NodeBatch) -> NodeOutput:
        if self.enable_nvtx:
            range_push(f"engine.ar.{batch.node_name}.{batch.graph_walk}")

        submodule = self.submodules.get(batch.node_name)
        if submodule is None:
            output = NodeOutput(
                per_request_output_tensors={rid: {} for rid in batch.request_ids}
            )
            if self.enable_nvtx:
                range_pop()
            return output

        try:
            # Priority: CUDA graph > batched > sequential
            if self._can_use_cuda_graph(batch):
                return self._execute_with_cuda_graph(batch, submodule)
            elif self._can_batch(batch):
                return self._execute_batched(batch, submodule)
            else:
                return self._execute_sequential(batch, submodule)
        finally:
            if self.enable_nvtx:
                range_pop()

    def add_request(
        self, request_id: str, cache_labels: list[str] | None = None,
    ) -> None:
        labels = cache_labels or ["main"]
        for label in labels:
            self.request_states[(request_id, label)] = KVRequestState()

    def remove_request(self, request_id: str) -> None:
        keys_to_remove = [
            k for k in self.request_states if k[0] == request_id
        ]
        for key in keys_to_remove:
            if self.page_allocator is not None:
                self.page_allocator.free(self.request_states[key].page_indices)
            del self.request_states[key]

    def pause_request(
        self, request_id: str, cache_label: str = "main",
    ) -> None:
        """For interleaved loop: mark as paused, keep KV pages allocated."""
        key = (request_id, cache_label)
        if key in self.request_states:
            self.request_states[key].is_paused = True

    def resume_request(
        self, request_id: str, cache_label: str = "main",
    ) -> None:
        """Resume from paused state for next LLM step in loop."""
        key = (request_id, cache_label)
        if key in self.request_states:
            self.request_states[key].is_paused = False

    def shutdown(self) -> None:
        self.kv_cache = None
        self.workspace_buffer = None
        self.request_states.clear()
