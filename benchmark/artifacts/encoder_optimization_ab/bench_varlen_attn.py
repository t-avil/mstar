#!/usr/bin/env python
"""Microbenchmark: varlen (block-diagonal) attention backends for the Qwen3-Omni
native vision encoder, on this H200 with NO flash-attn.

Vision attention is bidirectional, packed by cu_seqlens (one segment per image).
The current production fallback (_sdpa_varlen) materializes a dense
(total_tokens x total_tokens) mask -> O((B*L)^2). We compare it against backends
that keep cost O(B*L^2) (linear in batch):

  dense_mask   : current fallback (baseline)
  per_segment  : loop, dense SDPA per image segment
  padded_batch : pad segments to max_len, batched SDPA + key-padding mask
  flex         : torch FlexAttention with a block-diagonal mask_mod (compiled)
  nested       : NJT (nested tensor) SDPA varlen

Shapes from the real config: heads=16, head_dim=72. We sweep (batch B, per-image
tokens L) over representative single-image patch counts and serving batch sizes.
"""
import time, torch, torch.nn.functional as F

HEADS, HEAD_DIM, DT, DEV = 16, 72, torch.bfloat16, "cuda"
SCALE = HEAD_DIM ** -0.5


def make_qkv(seglens):
    total = sum(seglens)
    q = torch.randn(total, HEADS, HEAD_DIM, device=DEV, dtype=DT)
    k = torch.randn(total, HEADS, HEAD_DIM, device=DEV, dtype=DT)
    v = torch.randn(total, HEADS, HEAD_DIM, device=DEV, dtype=DT)
    cu = torch.tensor([0, *torch.tensor(seglens).cumsum(0).tolist()], device=DEV, dtype=torch.int32)
    return q, k, v, cu


def dense_mask(q, k, v, cu, scale):
    total = q.shape[0]
    seg = torch.zeros(total, dtype=torch.int32, device=q.device)
    seg[cu[1:-1].long()] = 1
    seg = torch.cumsum(seg, 0)
    m = seg[:, None] == seg[None, :]
    qb, kb, vb = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=m, scale=scale)
    return o.squeeze(0).transpose(0, 1).contiguous()


def per_segment(q, k, v, cu, scale):
    out = torch.empty_like(q)
    cul = cu.tolist()
    for a, b in zip(cul[:-1], cul[1:]):
        qs = q[a:b].transpose(0, 1).unsqueeze(0)
        ks = k[a:b].transpose(0, 1).unsqueeze(0)
        vs = v[a:b].transpose(0, 1).unsqueeze(0)
        o = F.scaled_dot_product_attention(qs, ks, vs, scale=scale)
        out[a:b] = o.squeeze(0).transpose(0, 1)
    return out


def padded_batch(q, k, v, cu, scale):
    cul = cu.tolist()
    lens = [b - a for a, b in zip(cul[:-1], cul[1:])]
    B, L = len(lens), max(lens)
    qb = q.new_zeros(B, L, HEADS, HEAD_DIM)
    kb = q.new_zeros(B, L, HEADS, HEAD_DIM)
    vb = q.new_zeros(B, L, HEADS, HEAD_DIM)
    mask = torch.zeros(B, 1, 1, L, device=q.device, dtype=torch.bool)
    for i, (a, b) in enumerate(zip(cul[:-1], cul[1:])):
        n = b - a
        qb[i, :n] = q[a:b]; kb[i, :n] = k[a:b]; vb[i, :n] = v[a:b]
        mask[i, 0, 0, :n] = True
    qb, kb, vb = (t.permute(0, 2, 1, 3) for t in (qb, kb, vb))
    o = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=mask, scale=scale)
    o = o.permute(0, 2, 1, 3)
    out = torch.empty_like(q)
    for i, (a, b) in enumerate(zip(cul[:-1], cul[1:])):
        out[a:b] = o[i, : b - a]
    return out


_flex = None
def flex(q, k, v, cu, scale):
    global _flex
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    total = q.shape[0]
    seg = torch.zeros(total, dtype=torch.int32, device=q.device)
    seg[cu[1:-1].long()] = 1
    seg = torch.cumsum(seg, 0)
    def mask_mod(b, h, qi, ki):
        return seg[qi] == seg[ki]
    bm = create_block_mask(mask_mod, None, None, total, total, device=q.device)
    if _flex is None:
        _flex = torch.compile(flex_attention)
    qb, kb, vb = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    o = _flex(qb, kb, vb, block_mask=bm, scale=scale)
    return o.squeeze(0).transpose(0, 1).contiguous()


def nested(q, k, v, cu, scale):
    cul = cu.tolist()
    lens = [b - a for a, b in zip(cul[:-1], cul[1:])]
    qs = [q[a:b] for a, b in zip(cul[:-1], cul[1:])]
    ks = [k[a:b] for a, b in zip(cul[:-1], cul[1:])]
    vs = [v[a:b] for a, b in zip(cul[:-1], cul[1:])]
    qn = torch.nested.nested_tensor([t.transpose(0, 1) for t in qs], layout=torch.jagged)
    kn = torch.nested.nested_tensor([t.transpose(0, 1) for t in ks], layout=torch.jagged)
    vn = torch.nested.nested_tensor([t.transpose(0, 1) for t in vs], layout=torch.jagged)
    o = F.scaled_dot_product_attention(qn, kn, vn, scale=scale)
    return o


BACKENDS = {"dense_mask": dense_mask, "per_segment": per_segment,
            "padded_batch": padded_batch, "flex": flex, "nested": nested}


def time_fn(fn, q, k, v, cu, iters=20):
    try:
        for _ in range(3):
            fn(q, k, v, cu, SCALE)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(q, k, v, cu, SCALE)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000.0
    except Exception as e:
        return f"ERR {type(e).__name__}: {str(e)[:60]}"


def main():
    print(f"device={torch.cuda.get_device_name()} torch={torch.__version__} dtype={DT}")
    # (label, seglens) — one image of L patches replicated B times (serving batch)
    cases = []
    for L in (576, 1024, 2304):       # small / medium / large single image
        for B in (1, 4, 8, 16, 32):
            cases.append((f"B={B:2d} L={L} (27 layers)", [L] * B))
    print(f"\n{'case':28s} " + " ".join(f"{b:>13s}" for b in BACKENDS))
    for label, seglens in cases:
        q, k, v, cu = make_qkv(seglens)
        row = []
        for name, fn in BACKENDS.items():
            r = time_fn(fn, q, k, v, cu)
            row.append(f"{r:13.3f}" if isinstance(r, float) else f"{r:>13s}")
        print(f"{label:28s} " + " ".join(row))
        del q, k, v, cu
        torch.cuda.empty_cache()
    print("\n(ms per single-layer attention call; x27 layers for full-encoder impact)")


if __name__ == "__main__":
    main()
