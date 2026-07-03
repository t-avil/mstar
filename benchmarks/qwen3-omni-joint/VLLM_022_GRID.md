# vLLM 0.22 vs our engine — what they changed, what we built, in plain language

Rewritten 2026-07-03 (v2) for readability. Every number traces to a committed
file. Status words are definitive: **SHIPPED** (on by default in our best
config), **VALIDATED-OPT-IN** (works, off by default), **REJECTED** (tried,
measured, lost), **WIP**, **BACKLOG** (not started).

**The one-paragraph story:** vLLM 0.22 got 3.2× faster at image→text high
concurrency, not because their GPU math got better (ours is faster per-kernel),
but because they deleted almost all the Python that runs *between* GPU steps.
Our engine runs that in-between glue in Python (~9–13 ms per step at batch 32);
theirs compiles it away (~2–4 ms). Everything below is one side or the other
of that fight.

---

## Scoreboard first (req/s ratio, us ÷ vLLM 0.22; >1 = we win)

| batch → | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| image→text | 0.94 | **0.82 ❌** | **0.88 ❌** | **1.16 ✅** | 1.03 | **0.85–0.88 ❌** (warm server 0.88) |
| audio→text | 1.14 | 0.83* | 0.89* | **1.17 ✅** | 1.26* | **1.11 ✅** |
| audio→speech | 2.63 | 2.65 | 2.46 | 2.12 | 2.49 | 2.26 — all ✅ |
| image→speech | 2.47 | 2.54 | 2.85 | 2.89 | 2.80 | 2.39 — all ✅ |

\* = measured on the previous build; not yet re-measured on the final stack.
Speech we win everywhere (their multi-process handoff costs them a file lock +
memory copies per audio chunk; ours passes pointers). Text at B2–B4 and i2t
B32 are the open fights.

---

## PART 1 — What vLLM changed in 0.22 (and whether we have an answer)

### 1.1 They compile the whole model step into one GPU program
**Plain:** a GPU is fast but must be *told* what to do by Python, command by
command. Telling it 10,000 small commands per step from Python is slow. Two
tricks fix this: "CUDA graphs" (record the whole step's commands once, replay
them as ONE command) and "torch.compile" (turn the Python model code into
optimized machine code). vLLM 0.22 finally got BOTH working together for this
model — a bug fix (their commit bb9f21d0) stopped the vision input from
breaking the compiler.
**Do we have it?** **YES (but different).** We had full CUDA graphs before
they did — our model step is one replay too. What we do NOT have: they also
compile the *scheduling glue around* the step; ours is plain Python threads.
That glue is exactly where our remaining gap lives.
**Our answer:** delete the glue by hand, piece by piece (Part 2, items 8–11).
**Status: SHIPPED (graphs) / ongoing (glue).**

### 1.2 They stopped waiting for the GPU to hand back sampled tokens
**Plain:** after the GPU picks the next token, the result must be copied
GPU→CPU. If the CPU *waits* for that copy every step, the GPU sits idle
meanwhile. vLLM 0.22 starts the copy in the background and only checks it
later ("async scheduling") — CPU and GPU overlap.
**Do we have it?** **NO — this is the biggest thing they have that we don't.**
We tested three cheap versions (E9/E10/W3-design); all flat, because back then
our bottleneck was elsewhere. Now that we moved the other work out (sidecar),
this is the top structural item.
**Status: BACKLOG, next big build (option V1). Est. +5–10% at i2t B32.**

### 1.3 They do bookkeeping once per BATCH, not once per request
**Plain:** with 32 requests in flight, any Python that runs "for each request"
runs 32× per step. Moving it to "once per step, for the whole batch" is a
32:1 saving. Their 0.22 rebase did this for expert-routing records.
**Do we have it?** **YES.** This is most of our campaign: batched token
prep, batched stop-check, batched message emit, memoized routing (Part 2,
items 3, 8, 9, 10).
**Status: SHIPPED.**

### 1.4 They fixed a bug that had put their MoE math on a slow kernel
**Plain:** the model's "mixture of experts" (MoE) layer is the most expensive
math. A config bug had routed it to a slow implementation; commit 3a7c7f14
routed it back.
**Do we have an equivalent problem?** **NO — ours is faster than theirs.** We
rewrote MoE in fp8 (8-bit numbers = half the memory traffic = 1.4–1.6× faster
kernel; +13% end-to-end). Their fix just restored their normal speed.
**Status: SHIPPED (our fp8 MoE).**

### 1.5 They removed accidental GPU↔CPU waits
**Plain:** copying a value GPU→CPU the naive way silently *stops the whole
GPU pipeline* each time. Six of those per step ruins you. They removed
several (commit 26967510).
**Do we have it?** **YES.** We found six such stalls per step inside our
sampler and killed them (config cache + change-detection, Part 2 item 10).
**Status: SHIPPED.**

### 1.6 They mix new-request processing into ongoing decode steps
**Plain:** when a new request arrives mid-stream, someone must process its
prompt ("prefill"). If decode pauses while that happens, throughput dips.
vLLM chops the prompt into chunks and rides them along with decode steps, so
new arrivals barely disturb the stream. This is also why their time-to-first-
token stays flat and ours grows with batch (theirs 0.16 s vs ours 0.43 s at
B32).
**Do we have it?** **YES (but different, and only partly winning).** We built
the same idea *inside* CUDA graphs (they drop to slower mode for mixed steps;
we stay captured — technically ahead of them). It wins at B1–B8, neutral at
B32. Their arrival-friendly *policy* on top (token budgets) is still to do.
**Status: SHIPPED (mechanism) / BACKLOG (policy = option V2, helps TTFT and
the losing B2–B4 cells).**

---

## PART 2 — What we built to fight back (chronological)

### 2.1 fp8 MoE — 8-bit expert math
**Plain:** the MoE layer is limited by how fast weights stream from memory.
Store them in 8-bit instead of 16-bit → half the bytes → nearly 2× the speed,
if you handle the precision carefully (we validated outputs are identical).
**Result:** kernel 1.44–1.61× faster, **+13% end-to-end** at i2t B8/B32, and
27 GB less GPU memory. **Status: SHIPPED.**

### 2.2 Fused router top-k
**Plain:** picking which experts each token goes to took 3 GPU operations;
one fused kernel does it in 1. Small but free. **Result:** ~0.5–0.8 ms/step
at B32. **Status: SHIPPED.**

### 2.3 Batched worker CPU cuts
**Plain:** replaced 32 tiny per-request CPU operations per step (token
copies, stop checks, embeddings) with 1 batched operation each.
**Result:** +13–22% i2t. **Status: SHIPPED.**

### 2.4 Encoder placement (your B4 — expanded)
**Plain:** we run on 2 GPUs. The image/audio *encoders* (turn pixels/audio
into tokens) must live on one of them. GPU-1 runs text generation (thinker);
GPU-2 runs speech generation (talker + vocoder). Whoever shares a GPU with
the encoders loses some speed to them.
1. Originally encoders sat with the thinker → text paths paid the tax.
2. We moved encoders to the speech GPU ("encoff" config): **text +9–31%,
   speech −4–22%.** No single placement wins everything — it's a real
   see-saw.
3. **Per your green light: we ship BOTH configs** — text-serving deployments
   use the text-optimized config, speech-serving use the speech-optimized
   one. Both yamls exist and are benchmarked. **Status: SHIPPED (two
   configs).** The fancier version (scheduler routes each request's encoder
   to whichever GPU is idler at that moment — one config that wins both) is
   **BACKLOG (option #23)**, deprioritized now that two configs are
   acceptable.

### 2.5 Inline + batched token emit
**Plain:** every generated token used to be written to a shared-memory file
and announced with its own message (32 messages/step). Now tokens ride inside
one combined message per step. **Result:** wash on image paths, **+11% on
audio→text B32** (highest message rate). **Status: SHIPPED.**

### 2.6 Denser CUDA-graph sizes (W7)
**Plain:** graphs are recorded for fixed batch sizes (1,2,4,8,16,32). A
23-request step must run the 32-size graph — 28% wasted math. We added 24 and
28. **Result:** +13% at B32 in triage. **Status: SHIPPED.**

### 2.7 Chunked prefill + mixed batches inside CUDA graphs (W5)
**Plain:** see 1.6 — our version of their arrival-mixing, but fully inside
recorded graphs. **Result:** +2–7% at B1–B8, neutral B32; every correctness
risk retired. **Status: SHIPPED.** (Split-attention upgrade measured 3.3×
faster mixed-step attention but converts to ~0% e2e at current mix volume —
**VALIDATED-OPT-IN**, will matter when option V2 raises mix volume.)

### 2.8 Slim emit
**Plain:** each token's message carried a re-serialized description of where
it came from — every token, every step. Now the description is sent once and
cached; later tokens send just values. **Result:** **+17–26% i2t B32** —
biggest single win after fp8. (First attempt was 5× SLOWER due to a
concurrent-mutation bug; fixed, then converted.) **Status: SHIPPED.**

### 2.9 Fast route
**Plain:** deciding "where do this step's outputs go" re-walked the same
graph every token. Answer never changes mid-stream → compute once, replay.
**Result:** **+7–10% i2t B32.** **Status: SHIPPED.**

### 2.10 Sampler config cache + batched stop-check
**Plain:** the sampler re-uploaded 6 small config arrays to the GPU every
step, each upload silently stalling the pipeline (see 1.5); and stop-checking
did a tiny GPU→CPU read per request. Cached the configs, batched the reads.
**Result:** +7–10% together at i2t B32 (as a pair). Today's isolation test on
the warm lab says the cache *alone* is ~0 on the current stack — the win
lives in the pair. **Status: SHIPPED (both on).**

### 2.11 Emit sidecar + second-pass trims (route2/slim2/fast-send)
**Plain:** Python only runs one thread at a time (the GIL). Message-building
work was stealing time from the scheduling thread. We exiled ALL per-token
message construction to a separate *process* (its own Python, own GIL) —
copied from vLLM's architecture. **Result:** **+9.2% at B32, +8.2% at B1**
(canonical pair, adjacent A/B, byte-identical output stream verified).
**Status: SHIPPED.**

### 2.12 Things we tried that LOST (kept for the record)
| What | Plain-language why it lost | Status |
|---|---|---|
| Side-thread prefill (E7) | the "helper" thread stole the GIL from decode — 10× collapse at B32 | REJECTED |
| FlashAttention-3 decode | measured slower than FlashInfer for this model's head layout | REJECTED |
| 3rd graph buffer slot | more VRAM pressure, no overlap gain: −19% | REJECTED |
| Faster GIL switching | more thread context switches on hot loops: −22% | REJECTED |
| Direct GPU token feed (E9) | the hop it removes isn't the binding cost — retested TODAY on the newest stack: still 0.99× | CLOSED (twice) |
| Two-step decode (E10) | same reason as E9 at the time; worth ONE retest after V1 lands | PARKED |
| Scheduler micro-cuts (SCHED_PACK, today) | skipping "is anyone else waiting?" checks delayed new-request admission — the checks are load-bearing: −5–7% | REJECTED |
| Single-chunk eager folding | starved decode occupancy ~10% | REJECTED |

---

## PART 3 — Today's measurement findings (they change how we read ALL numbers)

1. **Cross-GPU-socket penalty measured: −15.5% at B32.** Our benchmark
   harness used to pin the server to CPU socket 1 regardless of which GPUs it
   used — correct for GPUs 4–7, wrong for 0–3. Now auto-derived. All
   historical "pair 0,1" numbers were 7–15% understated.
2. **Cold-server penalty measured: first benchmark cell after boot reads
   −30%.** Steady state at i2t B32 is **7.0–7.3 req/s = 0.86–0.89× of vLLM**
   — our best honest number. Harness now warms every server with a discarded
   cell.
3. **Fast iteration lab built:** one warm server stays up; every flag flips
   at runtime; an A/B costs ~7 minutes instead of ~25. Boot itself dropped to
   ~3 min (OS cache).

---

## PART 4 — What's next, in order

1. **V1 async scheduling** (1.2 above) — the last structural thing they have
   and we don't. **WIP-next, 3–5 days, est. +5–10% at i2t B32.**
2. **V2 arrival-friendly chunk policy** (1.6) — attacks BOTH the B2–B4
   losses and time-to-first-token; our graph machinery is already built,
   this is scheduling policy only. **BACKLOG, 2–4 days.**
3. **Canonical re-sweep with the warm protocol** → official numbers refresh
   (expect the committed table to move up across the board). **1 day.**
4. Speech durability items (their per-chunk file-lock handoff has a fix
   sitting disabled in their tree — our 2.2–2.9× speech lead will not last
   forever by itself). **BACKLOG.**
