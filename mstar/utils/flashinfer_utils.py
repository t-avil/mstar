"""FlashInfer utility wrappers for batched paged attention.

Provides:
- run_rms_norm / run_attention: simple single-request helpers
- FlashInferPrefillWrapper: batched prefill with paged KV cache, optional CUDA graph mode
- FlashInferDecodeWrapper: batched decode with paged KV cache, optional CUDA graph mode

CUDA graph mode requires:
- Static buffer pointers passed at construction (qo_indptr_buf, paged_kv_indptr_buf, etc.)
- plan() updates values via .copy_() without reallocating
- The same wrapper object must be used during both capture and replay

Adapted from VoxServe's flashinfer_utils.py for our KV cache layout:
  [num_layers, max_pages, 2, page_size, num_kv_heads, head_dim]
(VoxServe uses [n_pages, 2, page_size, n_heads, head_dim] without layer dim.)
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)


def xqa_decode_enabled() -> bool:
    """Feature flag for the plan-free xqa/trtllm decode path.

    Default OFF. When set (``MSTAR_XQA_DECODE=1``) the batched decode wrapper
    routes its ``run()`` through ``flashinfer.decode.trtllm_batch_decode_with_kv_cache``
    (which dispatches to the xqa kernel on SM90) instead of the plan-based
    ``BatchDecodeWithPagedKVCacheWrapper``. Plan-based remains the default and
    the fallback. This is the foundation for in-graph multistep decode and
    mixed decode + bounded-prefill capture (device-resident ``seq_lens``, no
    host replan).
    """
    return os.environ.get("MSTAR_XQA_DECODE", "0") not in ("0", "", "false", "False")


def build_dense_block_tables(
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    page_size: int,
    device: torch.device,
    max_pages_per_seq: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Convert FlashInfer ragged paged-KV metadata into the dense page table +
    device ``seq_lens`` that the plan-free xqa/trtllm decode kernel expects.

    Ragged (plan-based) inputs, per request ``i`` with pages
    ``[indptr[i], indptr[i+1])`` in ``paged_kv_indices``:
      - ``num_pages_i  = indptr[i+1] - indptr[i]``
      - ``seq_len_i    = (num_pages_i - 1) * page_size + last_page_len_i``

    Dense outputs:
      - ``block_tables`` : int32 ``[bs, max_pages_per_seq]``, row ``i`` holds the
        physical page ids for request ``i`` left-packed, remainder zero-padded.
      - ``seq_lens``     : uint32 ``[bs]`` (device), kv length per request.
      - ``max_seq_len``  : python int = ``max_pages_per_seq * page_size``.

    All work is device-side gather/scatter (capturable). Inputs may be CPU;
    they are moved to ``device`` first.
    """
    indptr = paged_kv_indptr.to(device, non_blocking=True).to(torch.int64)
    indices = paged_kv_indices.to(device, non_blocking=True).to(torch.int64)
    last_page_len = paged_kv_last_page_len.to(device, non_blocking=True).to(torch.int64)

    bs = indptr.shape[0] - 1
    num_pages = indptr[1:] - indptr[:-1]                       # [bs]
    seq_lens = (num_pages - 1) * page_size + last_page_len     # [bs]

    if max_pages_per_seq is None:
        # eager path: size to the widest request this call
        max_pages_per_seq = int(num_pages.max().item()) if bs > 0 else 1
        max_pages_per_seq = max(max_pages_per_seq, 1)

    block_tables = torch.zeros(bs, max_pages_per_seq, dtype=torch.int32, device=device)
    # Scatter ragged indices into the dense rectangle: for global page slot g in
    # [indptr[i], indptr[i+1]) the (row, col) is (i, g - indptr[i]).
    total_pages = int(indptr[-1].item()) if bs > 0 else 0
    if total_pages > 0:
        arange = torch.arange(total_pages, device=device)
        seg = torch.repeat_interleave(
            torch.arange(bs, device=device), num_pages
        )                                                       # request id per page slot
        col = arange - indptr[:-1][seg]                         # column within row
        block_tables[seg, col] = indices[:total_pages].to(torch.int32)

    return block_tables, seq_lens.to(torch.uint32), max_pages_per_seq * page_size


# ---------------------------------------------------------------------------
# In-graph multistep decode (MSTAR_XQA_MULTISTEP) — Stage A helpers.
#
# These are the pure, CPU/meta-testable primitives that make a K-step decode
# capturable in ONE CUDA graph on top of the plan-free xqa decode path:
#   - xqa_multistep_k():        flag reader (K; 0/1 => off).
#   - multistep_write_locations(): derive the KV write slot for the current
#     token PURELY from the device seq_lens buffer (page-roll math). No host
#     indices, so it is valid inside a capture region.
#   - multistep_pages_needed():  how many dense block_table columns must hold
#     pre-allocated pages so K steps never allocate inside the graph.
#   - ingraph_greedy_token():    argmax matching the host greedy tie-break.
#   - trim_after_eos():          host-side stop handling (over-generate K, trim
#     everything after the first EOS before emit).
# See campaign_i2t/XQA_MULTISTEP_DESIGN.md for the full capture body.
# ---------------------------------------------------------------------------


def xqa_multistep_k() -> int:
    """Number of decode steps to fuse into one CUDA graph (``MSTAR_XQA_MULTISTEP``).

    Returns 0 when the feature is off (env unset, ``0``, ``1``, or unparseable) —
    0 and 1 both mean "single step per graph", i.e. the existing behavior, so the
    flag-OFF path stays byte-identical. Any ``K >= 2`` requests a K-step in-graph
    decode. Only meaningful when ``MSTAR_XQA_DECODE=1`` (needs the plan-free,
    device-``seq_lens`` xqa kernel); with the plan-based wrapper the per-step host
    replan cannot be captured, so multistep silently stays off.
    """
    if not xqa_decode_enabled():
        return 0
    try:
        k = int(os.environ.get("MSTAR_XQA_MULTISTEP", "0"))
    except (ValueError, TypeError):
        return 0
    return k if k >= 2 else 0


def multistep_write_locations(
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """KV write location for the token being processed at the CURRENT ``seq_lens``.

    ``seq_lens[r]`` is the kv length of request ``r`` **including** the token
    whose K,V is written this step (same convention as ``build_dense_block_tables``
    and the single-step wrapper, whose ``set_kv_cache`` writes at
    ``pos = last_page_len - 1``). The token's absolute index is therefore
    ``seq_lens - 1``; its page column is ``(seq_lens-1) // page_size`` and its
    in-page offset is ``(seq_lens-1) % page_size``. The physical page id is
    gathered from the pre-allocated dense ``block_tables``.

    Everything is a gather / integer div-mod on device tensors — no ``.item()``,
    no host indices — so this runs INSIDE the capture region. Advancing
    ``seq_lens += 1`` between steps rolls ``col`` to the next (pre-reserved)
    block-table column automatically when a page boundary is crossed.

    Args:
        seq_lens:     [bs] int/uint kv lengths (device or CPU for tests).
        block_tables: [bs, W] int32 dense page ids, left-packed, zero-padded.
        page_size:    tokens per KV page.
    Returns:
        locations: [bs, 2] int64 — column 0 = physical page id, column 1 =
        in-page offset. Same shape/semantics as the wrapper's
        ``kv_cache_locations``.
    """
    L = seq_lens.to(torch.int64)
    last = L - 1
    col = torch.div(last, page_size, rounding_mode="floor")        # [bs]
    off = last - col * page_size                                    # [bs]
    bs = block_tables.shape[0]
    rows = torch.arange(bs, device=block_tables.device)
    page = block_tables[rows, col].to(torch.int64)                 # gather
    return torch.stack([page, off], dim=1)


def multistep_pages_needed(
    seq_lens_before: torch.Tensor,
    k_steps: int,
    page_size: int,
) -> int:
    """Dense block-table columns that must hold valid pre-allocated pages to run
    ``k_steps`` in-graph decode steps starting from ``seq_lens_before``.

    Over the K steps the running length used at step ``i`` (0-based) is
    ``L0 + i`` and the token written that step has index ``L0 + i - 1``, so the
    widest page column touched (read for attention AND written) is
    ``(max(L0) + k_steps - 2) // page_size``. The block table (and the host
    paged allocator) must reserve **that many + 1** columns/pages up front, so no
    page allocation happens inside the graph.

    Returns the required column COUNT (== required pages per sequence, worst
    case). Callers size ``max_pages_per_seq`` (block-table width) to be ``>=``
    this and tell the paged allocator to pre-grow each sequence to
    ``max(L0) + k_steps - 1`` tokens before capture/replay.
    """
    L0 = int(seq_lens_before.max().item()) if seq_lens_before.numel() else 0
    max_idx = L0 + k_steps - 2           # last token index written across K steps
    if max_idx < 0:
        return 1
    return max_idx // page_size + 1


def ingraph_greedy_token(logits: torch.Tensor) -> torch.Tensor:
    """In-graph greedy token = ``argmax`` over the vocab dim.

    ``torch.argmax(dim=-1)`` returns the FIRST maximal index (lowest token id on
    ties), which matches the fused greedy kernel used by the single-step path
    (``fused_temperature_softmax(..., include_greedy=True)`` emits a one-hot at
    ``tl.argmax``, also first-index). Kept as a named helper so the multistep
    capture body and any parity test reference the exact same reduction.

    NOTE (parity): to be *byte-identical* to a mixed greedy/sampled batch the
    capture body should reuse M*'s ``Sampler``/``fused_temperature_softmax`` path
    directly (it already runs in-graph for the penalty case). This helper is the
    pure-greedy equivalent for the all-greedy benchmark config and for CPU tests;
    the CUDA tie-break of ``torch.argmax`` must be confirmed first-index on the
    live box (it is first-index on CPU).

    Args:
        logits: [bs, vocab] (any float dtype).
    Returns:
        tokens: [bs] int64.
    """
    return logits.argmax(dim=-1).to(torch.int64)


def trim_after_eos(
    tokens: torch.Tensor,
    eos_token_ids,
    already_finished=None,
    include_eos: bool = True,
) -> list[list[int]]:
    """Host-side stop handling for over-generated multistep output.

    The graph cannot early-stop, so it always emits ``K`` tokens per request.
    After the single ``[bs, K]`` host copy, drop everything a request produced
    AFTER its first stop token. This is done on the host BEFORE emit (never emit
    post-EOS), which avoids the ``async_sched`` trap of deferring the stop
    DECISION to a later step.

    Args:
        tokens:        [bs, K] the K in-order tokens sampled for each request.
        eos_token_ids: int or iterable of stop token ids.
        already_finished: optional [bs] bool — requests that hit EOS in a prior
                          batch emit nothing this batch.
        include_eos:    keep the EOS token itself (True) or drop it (False).
                        Single-step check_stop emits the stop token then stops,
                        so the default is True; confirm against M*'s check_stop.
    Returns:
        per-request list of emitted token ids (variable length <= K).
    """
    if isinstance(eos_token_ids, int):
        eos_set = {eos_token_ids}
    else:
        eos_set = set(int(x) for x in eos_token_ids)

    rows = tokens.tolist()
    bs = len(rows)
    fin = [False] * bs
    if already_finished is not None:
        fin = [bool(x) for x in (
            already_finished.tolist()
            if torch.is_tensor(already_finished) else already_finished
        )]

    out: list[list[int]] = []
    for r in range(bs):
        if fin[r]:
            out.append([])
            continue
        kept: list[int] = []
        for tok in rows[r]:
            tok = int(tok)
            if tok in eos_set:
                if include_eos:
                    kept.append(tok)
                break
            kept.append(tok)
        out.append(kept)
    return out


def multistep_keep_count(
    token_ids,
    eos_token_ids,
    ignore_eos: bool = False,
) -> int | None:
    """How many of the K over-generated multistep tokens to EMIT for one request.

    The K-step graph cannot early-stop, so it always samples K tokens. On the
    host, emit up to and including the FIRST stop token and drop the rest — the
    single-step "emit the stop token, then stop" contract expressed once per K.

    Args:
        token_ids:     the K in-order token ids for one request (list/iterable
                       of ints — the caller has already done the D->H copy that
                       check_stop makes, so this is pure host int logic).
        eos_token_ids: int or iterable of stop token ids.
        ignore_eos:    when True, never trim (keep all K).
    Returns:
        ``j + 1`` where ``j`` is the first stop-token index (keep [0..j]
        inclusive), or ``None`` to keep all K (no stop token this replay, or
        ``ignore_eos``). ``None`` (not ``K``) so callers can cheaply skip the
        slice in the common no-stop case.

    Kept as a pure named helper so the submodule trim hook and the CPU checks
    exercise the exact same first-EOS-inclusive logic as ``trim_after_eos``.
    """
    if ignore_eos:
        return None
    if isinstance(eos_token_ids, int):
        eos_set = {eos_token_ids}
    else:
        eos_set = {int(x) for x in eos_token_ids}
    for j, tok in enumerate(token_ids):
        if int(tok) in eos_set:
            return j + 1
    return None


@torch.compiler.disable
def run_rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-06,
    rms_norm_dtype=None
):
    orig_dtype = input.dtype
    if rms_norm_dtype is not None:
        input = input.to(rms_norm_dtype)
    elif torch.is_autocast_enabled():
        dtype = torch.get_autocast_gpu_dtype()
        input = input.to(dtype)
    elif input.dtype == torch.float32:
        # Unsupported dtype; must recast
        input = input.to(torch.bfloat16)

    # flashinfer.norm.rmsnorm requires matching input/weight dtypes; cast weight
    # to match whatever input ended up as.
    if weight.dtype != input.dtype:
        weight = weight.to(input.dtype)

    import flashinfer
    return flashinfer.norm.rmsnorm(
        input, weight, eps=eps
    ).to(orig_dtype)


def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float=1.0,
    causal: bool=True,
):
    import flashinfer
    return flashinfer.single_prefill_with_kv_cache(
        q,
        k,
        v,
        causal=causal,
        sm_scale=scale,
    )


class FlashInferPrefillWrapper:
    """Batched prefill attention with paged KV cache.

    Wraps flashinfer.BatchPrefillWithPagedKVCacheWrapper with:
    - Pre-computed token_to_page / token_to_cache for vectorized KV writes
    - Optional CUDA graph mode with static buffers

    Args:
        workspace_buffer: FlashInfer workspace (256MB+ recommended)
        num_qo_heads: number of query/output heads
        num_kv_heads: number of key/value heads
        head_dim: dimension per head
        page_size: KV cache page size
        batch_size: required for CUDA graph mode (max requests in batch)
        max_total_tokens: required for CUDA graph mode (max total new tokens across batch)
        max_num_pages: required for CUDA graph mode (max pages across all requests)
        device: torch device
        use_cuda_graph: if True, pre-allocate static buffers for graph capture
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        batch_size: int | None = None,
        max_total_tokens: int | None = None,
        max_num_pages: int | None = None,
        device: torch.device = torch.device("cuda"),
        use_cuda_graph: bool = False,
        enable_nvtx: bool = False,
    ):
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.enable_nvtx = enable_nvtx
        self.batch_size = batch_size
        self.max_total_tokens = max_total_tokens
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = None

        import flashinfer

        if self.use_cuda_graph:
            assert batch_size is not None, "batch_size required for CUDA graph mode"
            assert max_total_tokens is not None, "max_total_tokens required for CUDA graph mode"
            assert max_num_pages is not None, "max_num_pages required for CUDA graph mode"

            # Pre-allocate static index buffers
            self._qo_indptr_buf = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            )
            self._paged_kv_indptr_buf = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            )
            self._paged_kv_indices_buf = torch.zeros(
                max_num_pages, dtype=torch.int32, device=device
            )
            self._paged_kv_last_page_len_buf = torch.ones(
                batch_size, dtype=torch.int32, device=device
            )

            self.attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                qo_indptr_buf=self._qo_indptr_buf,
                paged_kv_indptr_buf=self._paged_kv_indptr_buf,
                paged_kv_indices_buf=self._paged_kv_indices_buf,
                paged_kv_last_page_len_buf=self._paged_kv_last_page_len_buf,
            )

            # Static buffers for vectorized KV cache writes
            self.token_to_page = torch.zeros(
                max_total_tokens, dtype=torch.long, device=device
            )
            self.token_to_cache = torch.zeros(
                max_total_tokens, dtype=torch.long, device=device
            )
        else:
            self.attn_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                workspace_buffer, "NHD"
            )
            self.token_to_page = None
            self.token_to_cache = None

        self._total_tokens = 0

    @torch.compiler.disable
    def plan(
        self,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        causal: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Plan attention and compute KV write indices.

        In CUDA graph mode, updates static buffers via .copy_() so that
        the same GPU addresses are used during graph replay.

        Inputs may be on CPU — that's preferred because FlashInfer's
        ``BatchPrefillWithPagedKVCacheWrapper.plan`` does ``indptr.to("cpu")``
        / ``last_page_len.to("cpu")`` internally; passing GPU tensors there
        triggers a synchronous default-stream sync that drains the
        speculatively-queued next decode step. We let the inner plan
        consume them as CPU and async-H2D copy to the device for our own
        per-token bookkeeping below.
        """
        self.dtype = dtype
        self.attn_wrapper.plan(
            qo_indptr=qo_indptr,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
            num_qo_heads=self.num_qo_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim_qk=self.head_dim,
            page_size=self.page_size,
            causal=causal,
            q_data_type=dtype,
        )

        # Async H2D for the GPU-side per-token bookkeeping that follows.
        if qo_indptr.device.type != "cuda":
            qo_indptr = qo_indptr.to(self.device, non_blocking=True)
            paged_kv_indptr = paged_kv_indptr.to(self.device, non_blocking=True)
            paged_kv_indices = paged_kv_indices.to(self.device, non_blocking=True)
            paged_kv_last_page_len = paged_kv_last_page_len.to(self.device, non_blocking=True)

        # Allow the qo_indptr to be accessible by BatchedCacheManager.get_qo_indptr_buf,
        # even if we're not in a cuda graph
        if not self.use_cuda_graph:
            self._qo_indptr_buf = qo_indptr

        # Compute per-token page and offset for vectorized KV writes
        n_req = qo_indptr.shape[0] - 1
        starts = qo_indptr[:-1].to(torch.int32)
        lens = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)
        total_tokens = int(lens.sum().item())
        self._total_tokens = total_tokens

        # Pages/lengths AFTER append
        num_pages_after = (
            paged_kv_indptr[1:] - paged_kv_indptr[:-1]
        ).to(torch.int32)
        kv_len_after = (
            (num_pages_after - 1) * self.page_size + paged_kv_last_page_len
        )

        # Flatten to per-token indices
        seg = torch.repeat_interleave(
            torch.arange(n_req, dtype=torch.int32, device=self.device), lens
        )
        intra = torch.arange(
            total_tokens, dtype=torch.int32, device=self.device
        ) - torch.repeat_interleave(starts, lens)

        # Absolute KV position per token
        start_new = kv_len_after[seg] - lens[seg]
        g = start_new + intra

        # Map to page + offset
        page_off = torch.div(g, self.page_size, rounding_mode="floor").to(
            torch.int32
        )
        off_in_page = (g - page_off * self.page_size).to(torch.int32)
        abs_page_ptr = paged_kv_indptr[:-1][seg] + page_off

        token_to_page = paged_kv_indices[abs_page_ptr].to(torch.long)
        token_to_cache = off_in_page.to(torch.long)

        if self.use_cuda_graph:
            self.token_to_page[:total_tokens].copy_(token_to_page)
            self.token_to_cache[:total_tokens].copy_(token_to_cache)
            if total_tokens < self.max_total_tokens:
                self.token_to_page[total_tokens:] = 0
                self.token_to_cache[total_tokens:] = 0
        else:
            self.token_to_page = token_to_page
            self.token_to_cache = token_to_cache

    @torch.compiler.disable
    def run(self, q: torch.Tensor, kv_cache_layer: torch.Tensor) -> torch.Tensor:
        """Run planned batched prefill attention.

        Args:
            q: [total_tokens, num_qo_heads, head_dim]
            kv_cache_layer: [max_pages, 2, page_size, num_kv_heads, head_dim]
                (single layer slice of the full KV cache)
        Returns:
            output: [total_tokens, num_qo_heads, head_dim]
        """
        return self.attn_wrapper.run(q.to(self.dtype), kv_cache_layer)

    def set_kv_cache(
        self,
        kv_cache_layer: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Write K, V to the paged KV cache at pre-computed positions.

        Args:
            kv_cache_layer: [max_pages, 2, page_size, num_kv_heads, head_dim]
            k: [total_tokens, num_kv_heads, head_dim]
            v: [total_tokens, num_kv_heads, head_dim]
        """
        n = self._total_tokens
        page_idx = self.token_to_page[:n]
        cache_idx = self.token_to_cache[:n]
        kv_cache_layer[page_idx, 0, cache_idx] = k[:n].to(self.dtype)
        kv_cache_layer[page_idx, 1, cache_idx] = v[:n].to(self.dtype)


class FlashInferDecodeWrapper:
    """Batched decode attention with paged KV cache.

    Optimized for the common decode case where each request appends
    exactly 1 new token. Uses BatchDecodeWithPagedKVCacheWrapper.

    Args:
        workspace_buffer: FlashInfer workspace
        num_qo_heads: number of query/output heads
        num_kv_heads: number of key/value heads
        head_dim: dimension per head
        page_size: KV cache page size
        batch_size: required for CUDA graph mode (max requests in batch)
        max_num_pages: required for CUDA graph mode (max pages across all requests)
        device: torch device
        use_cuda_graph: if True, pre-allocate static buffers for graph capture
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        batch_size: int | None = None,
        max_num_pages: int | None = None,
        device: torch.device = torch.device("cuda"),
        use_cuda_graph: bool = False,
        enable_nvtx: bool = False,
    ):
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.enable_nvtx = enable_nvtx
        self.batch_size = batch_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.dtype = None

        import flashinfer

        if self.use_cuda_graph:
            assert batch_size is not None, "batch_size required for CUDA graph mode"
            assert max_num_pages is not None, "max_num_pages required for CUDA graph mode"

            self._paged_kv_indptr_buf = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=device
            )
            self._paged_kv_indices_buf = torch.zeros(
                max_num_pages, dtype=torch.int32, device=device
            )
            self._paged_kv_last_page_len_buf = torch.ones(
                batch_size, dtype=torch.int32, device=device
            )

            self.attn_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                use_tensor_cores=True,
                paged_kv_indptr_buffer=self._paged_kv_indptr_buf,
                paged_kv_indices_buffer=self._paged_kv_indices_buf,
                paged_kv_last_page_len_buffer=self._paged_kv_last_page_len_buf,
            )

            # Static buffer for KV write locations: [batch_size, 2] = (page_idx, pos_idx)
            self.kv_cache_locations = torch.zeros(
                batch_size, 2, dtype=torch.long, device=device
            )
        else:
            self.attn_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                workspace_buffer, "NHD",
                use_tensor_cores=True,
            )
            self.kv_cache_locations = None

        # Flag-gated plan-free xqa decode path (default OFF). When enabled, the
        # plan-based wrapper above is still constructed (fallback / KV bookkeeping)
        # but run() is routed through the xqa kernel. See xqa_decode_enabled().
        self._xqa = None
        if xqa_decode_enabled():
            self._xqa = FlashInferXqaDecodeWrapper(
                workspace_buffer,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                batch_size=batch_size,
                max_num_pages=max_num_pages,
                max_pages_per_seq=max_num_pages if self.use_cuda_graph else None,
                device=device,
                use_cuda_graph=self.use_cuda_graph,
                enable_nvtx=enable_nvtx,
            )
            logger.info(
                "MSTAR_XQA_DECODE=1: batched decode run() routed through plan-free "
                "xqa/trtllm kernel (plan-based wrapper kept as fallback)."
            )

    def plan(
        self,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        kv_cache_locations: torch.Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Plan decode attention and compute KV write locations.

        For decode, each request appends exactly 1 token. The write
        location is the last page at position = last_page_len (before
        the append; after append it becomes last_page_len).

        Inputs may be on CPU; see prefill wrapper's plan docstring.
        """
        n_req = paged_kv_indptr.shape[0] - 1

        if self.enable_nvtx:
            from mstar.utils.profiler import range_pop, range_push

            range_push("flashinfer.decode.plan_inner", synchronize=False)
        try:
            self.attn_wrapper.plan(
                indptr=paged_kv_indptr,
                indices=paged_kv_indices,
                last_page_len=paged_kv_last_page_len,
                num_qo_heads=self.num_qo_heads,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
                page_size=self.page_size,
                q_data_type=dtype,
            )
        finally:
            if self.enable_nvtx:
                range_pop(synchronize=False)

        # Async H2D before our own per-rid bookkeeping.
        if paged_kv_indptr.device.type != "cuda":
            if self.enable_nvtx:
                range_push("flashinfer.decode.metadata_h2d", synchronize=False)
            try:
                paged_kv_indptr = paged_kv_indptr.to(self.device, non_blocking=True)
                paged_kv_indices = paged_kv_indices.to(self.device, non_blocking=True)
                paged_kv_last_page_len = paged_kv_last_page_len.to(self.device, non_blocking=True)
            finally:
                if self.enable_nvtx:
                    range_pop(synchronize=False)

        if kv_cache_locations is not None:
            locations = kv_cache_locations
            if locations.device.type != "cuda":
                if self.enable_nvtx:
                    range_push("flashinfer.decode.kv_location_h2d", synchronize=False)
                try:
                    locations = locations.to(self.device, non_blocking=True)
                finally:
                    if self.enable_nvtx:
                        range_pop(synchronize=False)
        else:
            # Compute KV write locations: page and position for each request's new token
            if self.enable_nvtx:
                range_push("flashinfer.decode.kv_location_compute", synchronize=False)
            try:
                page_idx = paged_kv_indices[paged_kv_indptr[1:] - 1]
                pos_idx = paged_kv_last_page_len - 1

                locations = torch.stack([page_idx.to(torch.long), pos_idx.to(torch.long)], dim=1)
            finally:
                if self.enable_nvtx:
                    range_pop(synchronize=False)

        if self.use_cuda_graph:
            if self.enable_nvtx:
                range_push("flashinfer.decode.kv_location_copy", synchronize=False)
            try:
                self.kv_cache_locations[:n_req].copy_(locations)
            finally:
                if self.enable_nvtx:
                    range_pop(synchronize=False)
        else:
            self.kv_cache_locations = locations

        self._n_req = n_req
        self.dtype = dtype

        # Mirror the plan into the xqa wrapper (builds dense block_tables +
        # device seq_lens; no host sync).
        if self._xqa is not None:
            self._xqa.plan(
                paged_kv_indptr=paged_kv_indptr,
                paged_kv_indices=paged_kv_indices,
                paged_kv_last_page_len=paged_kv_last_page_len,
                kv_cache_locations=kv_cache_locations,
                dtype=dtype,
            )

    @torch.compiler.disable
    def run(self, q: torch.Tensor, kv_cache_layer: torch.Tensor) -> torch.Tensor:
        """Run planned batched decode attention.

        Args:
            q: [n_req, num_qo_heads, head_dim]
            kv_cache_layer: [max_pages, 2, page_size, num_kv_heads, head_dim]
        Returns:
            output: [n_req, num_qo_heads, head_dim]
        """
        if self._xqa is not None:
            return self._xqa.run(q, kv_cache_layer)
        return self.attn_wrapper.run(q.to(self.dtype), kv_cache_layer)

    def set_kv_cache(
        self,
        kv_cache_layer: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Write K, V for decode (1 token per request).

        Args:
            kv_cache_layer: [max_pages, 2, page_size, num_kv_heads, head_dim]
            k: [n_req, num_kv_heads, head_dim]
            v: [n_req, num_kv_heads, head_dim]
        """
        # When the plan-free xqa path is active, route the KV write through the
        # xqa wrapper's OWN kv_cache_locations. This is byte-identical for
        # single-step (plan() sets both wrappers' locations from the same
        # ``locations`` tensor, so they are equal), but it is REQUIRED for
        # in-graph multistep: ``advance_step_ingraph`` advances only the xqa
        # wrapper's seq_lens + kv_cache_locations between the K captured
        # forwards, so the per-step KV write must read the xqa locations to land
        # at the correct (rolling) page/offset instead of a frozen step-0 slot.
        if self._xqa is not None:
            self._xqa.set_kv_cache(kv_cache_layer, k, v)
            return
        n = self._n_req
        pages = self.kv_cache_locations[:n, 0]
        positions = self.kv_cache_locations[:n, 1]
        kv_cache_layer[pages, 0, positions] = k[:n].to(self.dtype)
        kv_cache_layer[pages, 1, positions] = v[:n].to(self.dtype)


class FlashInferXqaDecodeWrapper:
    """Plan-free batched decode attention with paged KV cache (xqa / trtllm-gen).

    Drop-in replacement for :class:`FlashInferDecodeWrapper`'s ``plan``/``run``/
    ``set_kv_cache`` interface, backed by
    ``flashinfer.decode.trtllm_batch_decode_with_kv_cache`` (``backend="auto"``
    dispatches to the **xqa** kernel on SM90 / H200).

    Why: the plan-based ``BatchDecodeWithPagedKVCacheWrapper.plan`` host-syncs
    (``indptr.to("cpu")`` + ``int(max(kv_lens).item())`` at ``decode.py:1102``),
    which cannot appear inside a CUDA graph capture region, so N decode steps
    cannot be captured in one graph. The xqa entry point is plan-free: it takes a
    **device-resident** ``seq_lens`` (uint32) and a **dense** ``block_tables``
    (int32 ``[bs, max_pages_per_seq]``) directly, does no host sync, and is a
    registered custom op with a fake/meta impl — so it traces + captures cleanly,
    and ``seq_lens`` can be advanced in-graph between steps with no host replan.

    KV layout is M*'s native single-tensor NHD
    ``[num_pages, 2, page_size, num_kv_heads, head_dim]`` — no reshape needed.

    ``set_kv_cache`` and the write-location bookkeeping are copied verbatim from
    the plan-based wrapper so KV writes are byte-identical across the two paths;
    only the attention read kernel differs.
    """

    def __init__(
        self,
        workspace_buffer: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        batch_size: int | None = None,
        max_num_pages: int | None = None,
        max_pages_per_seq: int | None = None,
        device: torch.device = torch.device("cuda"),
        use_cuda_graph: bool = False,
        enable_nvtx: bool = False,
    ):
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self.enable_nvtx = enable_nvtx
        self.batch_size = batch_size
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.max_pages_per_seq = max_pages_per_seq
        self.dtype = None
        self.sm_scale = 1.0 / (head_dim ** 0.5)
        # In-graph multistep (MSTAR_XQA_MULTISTEP). K>=2 => the capture body runs
        # K decode iterations, advancing seq_lens + kv write-locations in-graph
        # between them. K in {0,1} => single-step (unchanged). Buffers allocated
        # lazily in plan()/enable_multistep() only when K>=2 and cuda-graph mode.
        self.multistep_k = xqa_multistep_k()
        self._tokens_out_buf: torch.Tensor | None = None

        # xqa views the workspace as uint8; first 8MB is the semaphore region and
        # MUST be zero on first use. Own a dedicated zeroed buffer (128MB) so we
        # never collide with the plan-based wrapper's workspace.
        self.workspace_buffer = torch.zeros(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )

        if self.use_cuda_graph:
            assert batch_size is not None, "batch_size required for CUDA graph mode"
            assert max_num_pages is not None, "max_num_pages required for CUDA graph mode"
            assert max_pages_per_seq is not None, (
                "max_pages_per_seq required for CUDA graph mode (dense page table width)"
            )
            # Static device buffers advanced in-place across plan() / in-graph steps.
            self._seq_lens_buf = torch.zeros(batch_size, dtype=torch.uint32, device=device)
            self._block_tables_buf = torch.zeros(
                batch_size, max_pages_per_seq, dtype=torch.int32, device=device
            )
            self.kv_cache_locations = torch.zeros(
                batch_size, 2, dtype=torch.long, device=device
            )
            # --- In-graph multistep advance scratch (persistent, in-place only) ---
            # The kernel's seq_lens is uint32, but torch has no CUDA add kernel for
            # uint32 ("ufunc_add_CUDA not implemented for 'UInt32'"), and the old
            # advance also *allocated* new tensors every step (torch.arange /
            # torch.stack / div-mod intermediates) inside the capture region. Both
            # are illegal-in-capture. So keep an int64 running-length as the
            # source of truth (int64 add IS supported), advance it in place, and
            # cast-copy it into the uint32 kernel buffer. All write-location math
            # writes into these pre-allocated buffers via out=/copy_ — ZERO
            # allocations inside advance_step_ingraph().
            self._ms_len = torch.zeros(batch_size, dtype=torch.int64, device=device)   # running seq_lens (int64)
            self._ms_last = torch.zeros(batch_size, dtype=torch.int64, device=device)  # token idx = len-1
            self._ms_col = torch.zeros(batch_size, dtype=torch.int64, device=device)   # page column
            self._ms_off = torch.zeros(batch_size, dtype=torch.int64, device=device)   # in-page offset
            self._ms_page = torch.zeros(batch_size, 1, dtype=torch.int32, device=device)  # gathered page id
        else:
            self._seq_lens_buf = None
            self._block_tables_buf = None
            self.kv_cache_locations = None
            self._ms_len = None
            self._ms_last = None
            self._ms_col = None
            self._ms_off = None
            self._ms_page = None

    def plan(
        self,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        kv_cache_locations: torch.Tensor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """Build the dense page table + device ``seq_lens`` (no host sync, no
        FlashInfer ``.plan()``). Mirrors ``FlashInferDecodeWrapper.plan`` args so
        it is a drop-in.
        """
        n_req = paged_kv_indptr.shape[0] - 1
        self.dtype = dtype
        self._n_req = n_req

        block_tables, seq_lens, max_seq_len = build_dense_block_tables(
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            self.page_size,
            self.device,
            max_pages_per_seq=self.max_pages_per_seq,
        )
        self._max_seq_len = max_seq_len

        if self.use_cuda_graph:
            self._seq_lens_buf[:n_req].copy_(seq_lens)
            self._block_tables_buf[:n_req, : block_tables.shape[1]].copy_(block_tables)
            self._seq_lens = self._seq_lens_buf[:n_req]
            self._block_tables = self._block_tables_buf[:n_req]
            # Seed the int64 running-length used by the in-graph multistep advance
            # so seq_lens += 1 works without a uint32 CUDA add (see advance_step_
            # ingraph). Runs at plan() time (outside capture): allocation is fine.
            self._ms_len[:n_req].copy_(seq_lens.to(torch.int64))
        else:
            self._seq_lens = seq_lens
            self._block_tables = block_tables

        # KV write locations (identical math to the plan-based wrapper).
        if kv_cache_locations is not None:
            locations = kv_cache_locations.to(self.device, non_blocking=True)
        else:
            indptr = paged_kv_indptr.to(self.device, non_blocking=True)
            indices = paged_kv_indices.to(self.device, non_blocking=True)
            last_page_len = paged_kv_last_page_len.to(self.device, non_blocking=True)
            page_idx = indices[indptr[1:] - 1]
            pos_idx = last_page_len - 1
            locations = torch.stack(
                [page_idx.to(torch.long), pos_idx.to(torch.long)], dim=1
            )
        if self.use_cuda_graph:
            self.kv_cache_locations[:n_req].copy_(locations)
        else:
            self.kv_cache_locations = locations

    @torch.compiler.disable
    def run(self, q: torch.Tensor, kv_cache_layer: torch.Tensor) -> torch.Tensor:
        """Run plan-free xqa/trtllm decode.

        Args:
            q: [n_req, num_qo_heads, head_dim]  (q_len_per_req == 1)
            kv_cache_layer: [max_pages, 2, page_size, num_kv_heads, head_dim] (NHD)
        Returns:
            output: [n_req, num_qo_heads, head_dim]
        """
        from flashinfer.decode import trtllm_batch_decode_with_kv_cache

        q = q.to(self.dtype)
        out = trtllm_batch_decode_with_kv_cache(
            query=q,
            kv_cache=kv_cache_layer,               # NHD [pages, 2, page, kv_heads, hd]
            workspace_buffer=self.workspace_buffer,
            block_tables=self._block_tables,
            seq_lens=self._seq_lens,
            max_seq_len=self._max_seq_len,
            bmm1_scale=self.sm_scale,              # == plan-based default sm_scale
            bmm2_scale=1.0,
            kv_layout="NHD",
            backend="auto",                        # SM90 -> xqa
            q_len_per_req=1,
        )
        return out

    def set_kv_cache(
        self,
        kv_cache_layer: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Write K, V for decode (1 token per request). Identical to the
        plan-based wrapper so KV state is bit-identical across paths."""
        n = self._n_req
        pages = self.kv_cache_locations[:n, 0]
        positions = self.kv_cache_locations[:n, 1]
        kv_cache_layer[pages, 0, positions] = k[:n].to(self.dtype)
        kv_cache_layer[pages, 1, positions] = v[:n].to(self.dtype)

    # ---- In-graph multistep primitives (capturable; used only when K>=2) ----

    def advance_step_ingraph(self) -> None:
        """Advance decode state by ONE token, entirely with device tensor ops
        (capturable — no ``.item()``, no host indices, no reallocation).

        Called between the K captured forward passes. After the step-``i``
        forward has consumed the current ``seq_lens`` (attention) and written the
        current token's K,V at ``kv_cache_locations``, this:
          1. ``seq_lens += 1`` — so step ``i+1``'s attention sees the new length;
          2. recomputes ``kv_cache_locations`` from the advanced ``seq_lens`` via
             :func:`multistep_write_locations`, rolling to the next PRE-ALLOCATED
             block-table column when a page boundary is crossed.

        Preconditions (enforced by the host at plan/reserve time, NOT here):
          - ``block_tables`` already holds valid pages for every column the K
            steps will touch (see :func:`multistep_pages_needed`);
          - ``_max_seq_len`` covers ``max(seq_lens) + K``.
        """
        assert self.use_cuda_graph, "multistep advance requires cuda-graph buffers"
        n = self._n_req
        p = self.page_size
        # Advance the int64 running-length in place (uint32 has no CUDA add
        # kernel), then cast-copy into the uint32 buffer the kernel reads. Same
        # persistent addresses at replay.
        self._ms_len[:n] += 1                                   # int64 in-place add
        self._seq_lens_buf[:n].copy_(self._ms_len[:n])          # int64 -> uint32 sync
        # Write-location page-roll math, IDENTICAL semantics to
        # multistep_write_locations() but writing ONLY into pre-allocated
        # buffers via out=/copy_ (no torch.arange / torch.stack / new tensors).
        #   last = len - 1 ; col = last // p ; off = last - col*p
        #   page = block_tables[row, col]  (gather)
        torch.sub(self._ms_len[:n], 1, out=self._ms_last[:n])
        torch.div(self._ms_last[:n], p, rounding_mode="floor", out=self._ms_col[:n])
        torch.mul(self._ms_col[:n], p, out=self._ms_off[:n])
        torch.sub(self._ms_last[:n], self._ms_off[:n], out=self._ms_off[:n])
        # page = gather along the page-column dim; index view (unsqueeze) is a
        # zero-alloc view, gather writes into the pre-allocated int32 buffer.
        torch.gather(
            self._block_tables_buf[:n], 1, self._ms_col[:n].unsqueeze(1),
            out=self._ms_page[:n],
        )
        # Scatter into the static [bs,2] locations buffer (col0=page, col1=off);
        # copy_ handles the int32->int64 / int64->int64 casts in place.
        self.kv_cache_locations[:n, 0].copy_(self._ms_page[:n, 0])
        self.kv_cache_locations[:n, 1].copy_(self._ms_off[:n])

    def enable_multistep(self, k_steps: int) -> None:
        """Allocate the ``[bs, K]`` static output-token buffer used to collect the
        K in-graph-sampled tokens for one batched host copy after replay."""
        assert self.use_cuda_graph and self.batch_size is not None
        self.multistep_k = k_steps
        self._tokens_out_buf = torch.zeros(
            self.batch_size, k_steps, dtype=torch.int64, device=self.device
        )

    def record_token_ingraph(self, step: int, token_ids: torch.Tensor) -> None:
        """Write step-``i`` sampled tokens into column ``i`` of the static output
        buffer (capturable copy). Read back with ONE host copy after replay."""
        assert self._tokens_out_buf is not None, "call enable_multistep() first"
        n = self._n_req
        self._tokens_out_buf[:n, step].copy_(token_ids[:n].to(torch.int64))
