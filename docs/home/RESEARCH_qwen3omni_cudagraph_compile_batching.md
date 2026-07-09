# Qwen3-Omni on M*: CUDA Graphs, Batching, and torch.compile — Presentation Brief

Consolidated research for presenting the encoder-optimization findings. Marks each
claim: **[CODE]** verified from source/commits, **[DATA]** from raw benchmark JSON,
**[DOC]** asserted in repo markdown only (not independently verified).

---

## 0. The one-sentence thesis
M* gets its **~2× throughput and 1.7–2.4× better, flatter ITL** by keeping every
Thinker/Talker step on a **pre-captured, fixed-shape CUDA graph**. That requires
**separating prefill from decode**, which is *exactly why* its image→text **TTFT
trails vLLM**. These are not independent knobs — they are the two ends of **one
trade-off dial**. The encoder work pushes M* as far along that dial as is safe
without an architectural rewrite.

---

## 1. Background: how M* executes (the "walk" model) [CODE]
- A **walk** is a named, CUDA-graph-capturable step type: `prefill_vision`,
  `prefill_text`, `prefill_audio`, `thinker_decode`, `talker_decode`, etc.
- The scheduler enforces **one walk-type per micro-batch** (`micro_scheduler.py`):
  every request in a step is the same kind, so the step has a **fixed shape** and
  replays a pre-captured graph.
- A CUDA graph records the exact kernel launch sequence once and **replays** it,
  removing per-kernel CPU dispatch (~30% per step at small batch). Hard
  requirement: **static shapes, no data-dependent Python** in the captured region.

This pre-existed the encoder work: Thinker/Talker decode (all bs 1–32), their
prefill (token-budget buckets with pad-up), and the Code2Wav vocoder were already
graph-captured. The Talker decode even unrolls the **full RVQ depth loop** in-graph
(no eager region). [CODE: DESIGN_cuda_graph.md, commit 843627648]

---

## 2. Mechanisms primer (for the talk)

**CUDA graph** — record-once/replay GPU kernel sequence; needs static shapes.
Wins on dispatch-bound small-batch steps. Can't capture Python control flow
(`.item()`, `.tolist()`, data-dependent branches).

**torch.compile (Inductor)** — fuses/optimizes the forward into compiled kernels.
- `dynamic=None` (default): **specializes per shape** → recompiles (multi-second)
  on each new audio length / image resolution mid-serving ("recompile storm").
- `dynamic=True`: **one shape-polymorphic artifact** → no recompiles on novel
  shapes. Slightly heavier first compile; per-op kernels can be marginally slower.

**FlashInfer varlen** — a ragged-prefill attention kernel that takes `cu_seqlens`
as a **tensor** and processes all variable-length sequences in **one kernel, no
Python branching**. This is what makes encoder attention **capture-safe**. The
default SDPA "adaptive" varlen loops in Python over `cu_seqlens` → **uncapturable**.

**Mixed walk / piggyback / continuous batching** — putting 1-token decode queries
*and* N-token prefill chunks for new requests in the **same** forward step
(vLLM-style). Lets a new request emit its first token almost immediately.

**Record-around-attention (vLLM's trick)** — capture one **shape-polymorphic**
recording for everything *except* attention (which runs flexible anyway). That one
artifact is valid for decode, prefill, **or mixed** steps. vLLM gets mixing AND
graph speed by recording around the only op that resists static shapes. M* uses
**whole-step** graphs tied to fixed shapes, so it does **not** have this. [DOC: STORY.md §6]

---

## 3. Where CUDA graphs apply in M* [CODE]
| Component | Captured? | Keying |
|---|---|---|
| Thinker decode | ✅ full graph | bs {1,2,4,8,16,32} |
| Thinker prefill (text/audio/vision) | ✅ graph + pad-up | token buckets (…→2048 text/audio, …→16384 vision) |
| Talker decode (incl. full RVQ depth loop) | ✅ full graph | bs {1,2,4,8,16,32} |
| Code2Wav vocoder | ✅ full graph | bs {1,2,4,8,16,32} |
| **Native vision/audio encoder block loop** | ✅ **(new)** lazy per-layout, ≤16 keys | `(seq_len, tuple(cu_seqlens))` |
| Encoder CNN frontend | ❌ eager | `.item()/.tolist()` = data-dependent |
| Mixed (prefill+decode) step | ❌ eager | no matching fixed-shape graph |
| Prefill beyond top bucket ("eager cliff") | ❌ eager | >2048 text/audio tokens |

**Why encoders are harder than decode:** decode varies only in batch size (small
enumerable set). Encoder prefill varies in **both** total token count **and** the
per-sequence length distribution (`cu_seqlens`) → potentially unbounded shape
space. Solution: **lazy per-layout capture** keyed on `(tokens, cu_seqlens)`, capped
at 16 cached graphs; a never-repeating shape just runs eager (no recompile storm). [CODE]

---

## 4. What actually shipped (M*-new = "combined-lowrisk", `1f66ce6`) [CODE]
Only demonstrably-safe, no-regression, no-scheduler-change optimizations:
- Native encoders (replace HF wrappers).
- Encoder block-loop **CUDA graph** ON (`MSTAR_ENCODER_CUDA_GRAPH=1`), enabled by a
  3-default flip: varlen→`flashinfer`, CUDA_GRAPH→1, CG_WARMUP→`1,2,4,8`. [commit 8091d30]
- **`torch.compile(dynamic=True)`** on encoder forward — *tail-latency only*, see §7. [commit 1f66ce6]
- Vision **GPU→CPU sync elimination** in prefill prepare (grid_thw kept on CPU). [commit fbc9804]
- GPU mel extraction, GPU image preprocess, batched vision prefill, vision graph-align.

**Deliberately excluded** (and why): mixed-walk (20× collapse, §6), async-encoder
("flaky in integration testing", commit 95290f6), chunked-prefill (scheduler
re-enqueue is a `NotImplementedError` stub), wider prefill buckets / encoder-gap
(need GPU validation).

---

## 5. The headline result vs vLLM (i2t) [DATA: raw_image_to_text.json]
| B | M*-new TTFT | vLLM TTFT | gap | M*-new req/s | vLLM req/s | M*-new ITL | vLLM ITL |
|---|---|---|---|---|---|---|---|
| 1 | 0.215s | 0.150s | vLLM 1.43× | 0.755 | 0.373 (M* 2.0×) | 0.0068 | 0.0123 (M* 1.8×) |
| 8 | 0.344s | 0.187s | vLLM 1.84× | 2.52 | 1.02 (M* 2.5×) | 0.0151 | 0.0360 (M* 2.4×) |
| 32| 0.576s | 0.206s | vLLM 2.80× | 4.46 | 2.55 (M* 1.75×)| 0.0341 | 0.0574 (M* 1.68×)|

- TTFT gap exists at **every** batch and **widens with load** (prefill-queue signature).
- M* wins throughput **and** ITL at every batch; ITL stays flat while vLLM's degrades.
- M*-new already improved i2t TTFT **~1.87×** vs M*-old (B=1) — real progress; the
  *residual* gap to vLLM is structural.
- **Contrast (S2T):** M* *beats* vLLM TTFT at B=1 (0.097 vs 0.140) — GPU mel +
  in-process IPC win where there's no queue. i2t has no equivalent cheap win (the
  vision encoder is intrinsically large/variable).

---

## 6. Batching: the root cause of the TTFT gap
**M* separates prefill and decode** so every step stays on its captured graph. An
i2t request must traverse `prefill_vision → prefill_text → first decode` before
token 1; a new request waits for a prefill slot → TTFT grows with the queue. [DOC/CODE]

**vLLM mixes** prefill into decode steps (continuous batching) via
record-around-attention → a new request piggybacks onto the next decode step →
**flat TTFT under load**. [DOC]

**Why M* can't just mix:** a mixed step has shape `(decode_count, prefill_len)` that
matches **neither** the decode bucket **nor** the prefill bucket → it runs **eager**
on every mixed step. Measured: forcing mixing without a mixed graph
(`MSTAR_MIXED_WALK=1`, eager) at B=32 i2t → **0.19 vs 3.74 req/s, a 20× collapse**,
3 requests errored. [DATA/DOC: KNOWLEDGE §4]

---

## 7. torch.compile findings [CODE + DOC]
- Applied to encoder `forward` (`fullgraph=False`). With `dynamic=True` (final).
- **Does NOT improve steady-state throughput** — your microbench: *"torch.compile
  does NOT win vs eager for these encoders; the big matmuls dominate and Inductor
  can't beat them."* [DOC: KNOWLEDGE §1a — note: the microbench file itself could
  not be re-read, so cite as documented, not freshly measured]
- **Sole value = tail latency:** `dynamic=True` removes multi-second recompile
  stalls on novel audio lengths / image resolutions.
- Because the encoder is matmul-bound, already CUDA-graphed, and attention is on
  FlashInfer (`@torch.compiler.disable`d), compile touches only minor norm/elementwise
  glue → steady-state impact ≈ neutral either way.
- **Determinism caveat for demos:** any compile/CUDA-graph change is perf, not
  numerics — but the downstream **audio is nondeterministic regardless** (the Talker
  samples codec tokens; verified: same code + same input → different audio, even
  different length, run-to-run). Audio parity is therefore the wrong test; encoder
  parity is checked on the **embedding tensors** (cos > 0.9999). [CODE/measured here]

---

## 8. What was tried → what happened (the "we explored it" table)
| Approach | Result | Why not shipped |
|---|---|---|
| `MSTAR_MIXED_WALK` (vLLM-style mixing, eager) | B=32 i2t **0.19 vs 3.74 req/s, 20× collapse**, 3 errors | mixed shape matches no graph → all-eager [DATA] |
| mixed-CG **bucketed** (48 graphs) / **coarse** (6) | graphs captured, but `run_mixed` returned a raw dict not `NodeOutput` → **AttributeError every mixed step**; full grid warmup **>15 min** | correctness bug + prohibitive warmup; **B=32 post-fix PENDING** [DOC] |
| mixed-CG **supergraph** / **decode-only** | captured under wrong key → **silent eager fallback** (were just the eager baseline) | graph-key bug; no real mixed graph ran [DOC] |
| `MSTAR_ENCODER_ASYNC` (overlap encode w/ decode) | i2t B=32: **−30% TTFT, +7.4% req/s** (promising) | "flaky in integration testing" → default flipped OFF (commit 95290f6) [CODE] |
| `MSTAR_CHUNKED_PREFILL` | design + parity argument done | scheduler re-enqueue is a `NotImplementedError` stub; never GPU-validated [CODE] |
| `MSTAR_ENCODER_COALESCE` | S2T B=8 **−32% TTFT** (clean audio win) | audio-only; adds scheduler complexity → out of lowrisk [DOC] |
| wider prefill buckets / encoder-gap | designed (push eager cliff 2048→4096; shrink ~0.8ms→~0.07ms encoder→Thinker gap) | need GPU validation (VRAM / sync-elim) [CODE] |

**The 4 mixed-CG branches all point to one commit (6bdcb90); the experiment code
was never committed** — bug/warmup details are from `KNOWLEDGE_*.md` only. [DOC]

---

## 9. "As good as it gets" — the defensible close
1. The TTFT gap is **structural, identified, and quantified**, not an oversight.
2. Closing it has exactly three options: **(a)** mix as-is → 20× throughput collapse
   (demonstrated); **(b)** capture mixed-shape graphs → buggy + >15 min warmup, no
   clean high-load result; **(c)** adopt vLLM's record-around-attention substrate →
   a real engineering rewrite, not a flag.
3. M* takes the **throughput + flat-ITL wins now** (the metrics that drive serving
   cost and steady-state UX) and leaves the substrate migration as a deliberate
   future decision. On speech-out paths (S2S/I2S) M* also **beats** vLLM on
   TTFT-audio 1.9–2.6×.
4. That is a considered position backed by the full experiment record — say it
   plainly: *"we know why, we measured the cost of the fix, we chose the safe wins."*

---

## 10. Likely quiz questions + crisp answers
**Q: Why is i2t TTFT worse than vLLM, even at B=1?** ~65 ms of serialized
`prefill_vision → prefill_text → decode` before token 1; vLLM admits into the next
decode step immediately. No queue at B=1, so it's pure scheduling structure.

**Q: Why does the gap widen with load (1.43×→2.80×)?** vLLM continuous-batches new
prefills onto decode steps (flat TTFT); M* queues prefill slots. Widening = queuing.

**Q: Why not just mix like vLLM?** A mixed step's shape matches no captured graph →
eager every step. Measured 20× throughput collapse at B=32. Mixing *and* staying
fast needs record-around-attention — a substrate rewrite.

**Q: Did mixed-CG fix it?** No clean result: 2/4 variants silently ran eager (wrong
key), 2/4 crashed on a `NodeOutput` type bug; full grid warmup >15 min; **B=32
post-fix is still PENDING.** (Don't claim it fails — say it's open.)

**Q: What did torch.compile buy?** Not throughput (microbench: doesn't beat eager) —
only elimination of recompile stalls on novel shapes, via `dynamic=True`.

**Q: Why FlashInfer for the encoder graph?** SDPA "adaptive" varlen loops in Python
over `cu_seqlens` (uncapturable); FlashInfer takes `cu_seqlens` as a tensor, one
kernel, no branching → capture-safe.

**Q: What is the encoder CG keyed on, and won't it explode?** `(total_tokens,
tuple(cu_seqlens))`; capped at 16 cached graphs; novel layouts run eager. No storm.

**Q: Why is the Talker decode fully in-graph?** It captures the entire RVQ depth
loop (`forward_depth_unrolled`) with dense preallocated KV and static positions — no
eager region in that step.

**Q: Why does M* win S2T TTFT at B=1 but lose i2t?** S2T B=1 is bottlenecked by mel
extraction (GPU in M*, CPU in vLLM) + in-process IPC — M* wins. i2t has no cheap
win: the vision encoder is intrinsically large/variable.

**Q: Is the i2t TTFT regression acceptable?** Yes — structural + bounded; M* wins
throughput (~2×) and ITL (1.7–2.4×, flat under load while vLLM's degrades ~5×); i2t
TTFT already improved 1.87× vs M*-old; speech-out paths beat vLLM on TTFT-audio.

---

## 11. Don't over-claim (verification gaps)
- "torch.compile doesn't win" microbench file couldn't be re-read → cite as
  documented, not freshly measured.
- mixed-CG experiment code never committed (all 4 branches = commit 6bdcb90) → bug
  and warmup figures are from the knowledge doc only.
- **B=32 bucketed-mixed-CG post-fix result is PENDING** — the core open question
  (does graphed mixing recover throughput?) is unanswered.
- encoder-gap ~10× gap-reduction and ~15 min warmup are estimates, not measured.
- vLLM "record-around-attention" is from vLLM docs, not traced into vLLM source.
- The ~3.3 s/call HF vision-encoder cost is an in-code comment, not a standalone
  benchmark (the *delta* M*-old 0.401 → M*-new 0.215 TTFT is real [DATA]).

---

*Sources: `KNOWLEDGE_cudagraph_compile_mixedwalk.md`, `STORY.md`, `ARCHITECTURE.md`,
`NUMBERS.md`, `raw_*.json` on `pr/native-encoders-benchmarks-v3`; encoder source on
`pr/native-encoders-v3`; commits 8091d30, fbc9804, 1f66ce6, 95290f6, 843627648,
6bdcb90; opt/combined-lowrisk history.*
