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

=== NEW: Executive summary (for the manager) ===

**What changed between vLLM-Omni 0.21 and 0.22.** They rebased their
Qwen3-Omni pipeline onto the current vLLM V1 engine and fixed the four things
that had been keeping this model off vLLM's fast path: a vision-input bug
that silently disabled torch.compile (so in 0.22 the whole text decoder runs
as compiled code plus one full CUDA graph per step), the sampled-token
GPU→CPU copy became asynchronous (it now overlaps the next forward pass
instead of stalling it), per-request expert-routing bookkeeping became
per-batch, and a configuration bug that had routed their MoE layer to a slow
kernel was corrected. None of this is a new invention — it is vLLM's standard
machinery finally working for this model. The net effect: their per-step CPU
overhead dropped to roughly 2–4 ms and image→text at batch 32 went from ~2.5
to 8.2 req/s (3.2×). Their GPU kernels did not get better than ours.

**What we did about it.** Our campaign was also host-side. On the GPU we
quantized the MoE expert weights to block-fp8 (1.4–1.6× on the kernel, +13%
end-to-end), moved the image/audio encoders off the text-generation GPU
(+9–31% on text paths; shipped as two per-workload configs per your call),
added denser CUDA-graph batch sizes (24, 28) to cut padding waste, and built
chunked prefill plus mixed prefill/decode batches that stay inside captured
CUDA graphs (vLLM drops to a slower mode for those steps; we don't). On the
CPU we systematically removed per-token Python from the two hot threads:
tokens now travel in one batched message per step instead of 32; each
request's routing decision and its serialized message header are computed
once and replayed instead of rebuilt every token; six hidden GPU-pipeline
stalls in the sampler were cached away; and finally all per-token message
assembly was exiled into a separate "sidecar" process with its own Python
interpreter (+9% at B32, modeled on vLLM's own process architecture). Net:
image→text B32 moved from 0.53× to 0.86–0.90× of vLLM, B8/B16 now win
outright, audio→text is ≥1.1× at high batch, and speech remains 2.2–2.9×.
The remaining gap is one structural item — asynchronous sampled-token
copy-back, now in implementation — plus admission policy for small batches.

=== END NEW ===

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

=== EXPANDED: what "glue" means, why torch.compile can't fix it for us, what's left ===
**What the glue actually is:** everything the CPU must do between two GPU
steps: collect the tokens the GPU just picked, check each request for "am I
done?" (EOS / length limit), package tokens into messages for the API server,
decide which requests run in the next step, write each request's new token
into the next step's input buffer, and update per-request bookkeeping
(position counters, memory pages, loop state). vLLM does this in compact,
compiled code inside one process. Our engine is a general *graph-walking
framework*: it models the whole model pipeline as a graph of nodes and
literally walks that graph in Python for every request every step — flexible
(it's why we can serve thinker+talker+vocoder pipelines at all), but each walk
is dictionary lookups and Python object churn, times 32 requests, times every
step.
**Why we can't just torch.compile it like they did:** torch.compile
accelerates *tensor math* — it traces sequences of GPU operations and fuses
them. Our glue contains almost no tensor math; it is control flow, Python
dictionaries, and cross-process messaging. A tensor compiler cannot trace
that. vLLM's advantage isn't that they compiled their scheduler — it's that
their scheduler was already thin, hand-written code with the model-facing
parts compiled. (Rewriting our framework in C++/Rust was evaluated and
rejected: weeks of work, and the measured wins below got most of the value
in days.)
**What's left of the glue after this campaign:** roughly 4–9 ms/step at B32,
in three chunks: (1) the scheduler's "who runs next" scan and per-request
loop-state updates — measured to be load-bearing (our attempts to batch or
skip them regressed or were flat: SCHED_PACK, W2); (2) residual per-request
routing/bookkeeping — mostly memoized already; (3) the one remaining
*synchronous wait*: the CPU waits for the GPU's sampled tokens before
finishing each step's bookkeeping — that is exactly item 1.2 below, now in
implementation.
=== END EXPANDED ===

### 1.2 They stopped waiting for the GPU to hand back sampled tokens
**Plain:** after the GPU picks the next token, the result must be copied
GPU→CPU. If the CPU *waits* for that copy every step, the GPU sits idle
meanwhile. vLLM 0.22 starts the copy in the background and only checks it
later ("async scheduling") — CPU and GPU overlap.
**Do we have it?** **NO — this is the biggest thing they have that we don't.**
We tested three cheap versions (E9/E10/W3-design); all flat, because back then
our bottleneck was elsewhere. Now that we moved the other work out (sidecar),
this is the top structural item.
**Status: === WIP — implementation started today (option V1). === Est. +5–10% at i2t B32.**

=== EXPANDED: what moved to the sidecar, what exactly we tried before, how V1 differs ===
**What the sidecar took off the hot thread (item 2.11):** building and
sending every token's client-bound message, and assembling the
"request-finished-this-stage" notifications the coordinator uses for
accounting. All of that serialization and socket work used to run on the same
Python thread that schedules GPU steps; now a separate process does it.
**What we tried before and why each was flat:**
- *E9 (direct token feed):* fed the GPU's sampled tokens straight into the
  next step's input on the GPU, skipping a CPU round-trip. Flat — because the
  CPU round-trip it removed was running in parallel with other CPU work
  anyway; removing it just exposed the next wait.
- *E10 (two-step decode):* ran two GPU steps per scheduling round to halve
  scheduling overhead. Flat — same reason: scheduling overhead already
  overlapped GPU time.
- *W3 (run-ahead scheduling, from SGLang):* prepare step N+1's metadata
  before step N's tokens exist. Skipped after analysis — our speculation
  system already achieves that overlap.
All three attacked waits while the real cost was *work* (the main-thread
Python), which is why the sidecar had to come first.
**How V1 is different:** it removes the *last synchronous wait*. Today the
worker submits GPU step N, then **blocks until N's sampled tokens are copied
back to the CPU** before finishing N's bookkeeping and stop-checks. V1: the
token copy-back starts on a separate GPU channel; the worker *immediately*
submits step N+1 (its input tokens are already on the GPU — that's the E9
machinery, already validated); the copied-back tokens are consumed **one step
late** for stop-checks and message emission. Cost of the trade: a request
that finishes generates one extra throwaway token (bounded, and vLLM makes
the same trade). Result: the GPU never idles waiting for the CPU to read
tokens back. This is precisely the mechanism vLLM 0.22 added.
=== END EXPANDED ===

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

=== EXPANDED: what exactly we do today vs their policy, and the history ===
**What our latest build actually does — both pieces, to be precise:**
(1) *chunked prefill*: a new prompt is split into 256-token chunks instead of
being processed in one long serialized pass; (2) *mixed batching*: a chunk
can ride INSIDE the same captured CUDA-graph step as the ongoing decode
requests (packed side by side in one GPU launch). Both are on in the shipping
config.
**What we lack — the admission policy:** vLLM's scheduler gives every step a
token budget (e.g. 8192 tokens). Ongoing decodes claim their seats first;
whatever budget remains pulls in prompt chunks — *every single step,
continuously*. Ours only folds a chunk in at specific scheduler boundaries
(when the speculation chain yields), so under continuous arrivals a new
request waits longer for its first chunk and prompts drain more slowly. Same
engine capability, less aggressive usage of it.
**Is this why we lose TTFT and some throughput cells? Mostly yes, plus
history you're remembering correctly:** in vLLM-**0.21**, mixing prompts into
the stream knocked their decode off CUDA graphs into a slow eager mode — that
is a big part of why we won B32 throughput back then (their loss, not just
our win). The 0.22 rebase fixed exactly that: they now keep full speed while
absorbing arrivals every step. That collapsed the old trade ("they win TTFT,
we win throughput") into "they win TTFT everywhere (0.16 s vs 0.43 s at B32)
and win throughput at B2–B4 and B32". Our TTFT also carries a second,
unrelated tax: our prompt path crosses multiple processes (encoder walk →
coordinator round-trips) before decoding starts. V2 (the token-budget policy
on our existing machinery) attacks the first cause; the multi-process prompt
path is a separate, known cost.
=== END EXPANDED ===

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

=== CLARIFIED: this is decode-only, nothing to do with prefill/mixed ===
This has nothing to do with prefill or mixed batching. It's about ordinary
**decode** steps: you serve 32 requests, but they finish at different times,
so at any moment only, say, 23 are still generating. A CUDA graph is a
recording for one FIXED batch size — you can't run a "23-row" step unless a
23-size recording exists. Before, the nearest recording ≥23 was 32, so the
step ran the 32-slot recording with 9 slots padded with dummy rows: the GPU
computes 32 rows of math and throws 9 away (~28% waste). We recorded two
extra sizes (24 and 28) so a 23-request step now runs the 24-recording —
~4% waste instead of 28%.
=== END CLARIFIED ===

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

=== CLARIFIED: no, it wasn't tracing ===
This was not tracing or telemetry — it was **protocol overhead in the token
delivery path itself**. Every generated token must reach the API server
inside a message so it can be streamed to the client. Each such message
included a full serialized copy of the token's routing header: which request
it belongs to, which output edge of the model graph produced it, and the
stream-position object needed to slot it into the response — a pickled Python
object that is IDENTICAL for every token of a given request. We now send that
header once (with the first token); the API server caches it; every later
token sends just the raw values plus a small integer key. Same information
delivered, ~all of the per-token serialization cost gone.
=== END CLARIFIED ===

### 2.9 Fast route
**Plain:** deciding "where do this step's outputs go" re-walked the same
graph every token. Answer never changes mid-stream → compute once, replay.
**Result:** **+7–10% i2t B32.** **Status: SHIPPED.**

=== EXPANDED: what this means technically ===
Our engine represents the model pipeline as a graph. After every step, for
every request, the worker must route the step's outputs: this token tensor
loops back as the next step's input; this one goes out to the API server;
on speech paths, this one crosses to the other GPU's worker. Before, the
worker re-derived that routing from scratch each token — walking the node's
outgoing edges, classifying each (loop-back / client / cross-worker /
persist), and building fresh Python routing objects and fan-out lists — even
though for a decode loop the answer is identical every single step.
FAST_ROUTE caches the computed routing plan per (request, node) after the
first token and replays it, re-cloning only the fields that genuinely change
(tensor addresses). Honest footnote from today's re-measurement: after the
sidecar landed, FAST_ROUTE's standalone contribution pooled to ~0–3% (the
sidecar removed neighboring work, shrinking what the cache saves); it stays
on.
=== END EXPANDED ===

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

=== CLARIFIED: what "message" means here ===
Two kinds of messages, both previously built on the worker's scheduling
thread every step: (1) the **token-delivery message** — the packet carrying
each newly generated token (plus its routing key, see 2.8) to the API server
so it can stream text back to the client; (2) the **completion notifications**
— the records telling the coordinator "request X finished node Y", which
drive request-lifecycle accounting. Assembling these means creating Python
objects, serializing them, and pushing them into sockets — thousands of times
per second at batch 32, all while holding the same interpreter lock the GPU
scheduler needs. The sidecar is a separate operating-system process: the
worker now hands it raw token data through a lock-free queue, and ALL object
assembly, serialization, and socket work happens in the sidecar's own Python
interpreter. Two processes = two GILs = the scheduler thread never waits on
messaging again. A committed byte-identity test proves the client-visible
stream is unchanged.
=== END CLARIFIED ===

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
   and we don't. === **WIP — implementation started 2026-07-03 evening** ===, est. +5–10% at i2t B32.
2. **V2 arrival-friendly chunk policy** (1.6) — attacks BOTH the B2–B4
   losses and time-to-first-token; our graph machinery is already built,
   this is scheduling policy only. **BACKLOG, 2–4 days.**
3. **Canonical re-sweep with the warm protocol** → official numbers refresh
   (expect the committed table to move up across the board). **1 day.**
4. Speech durability items (their per-chunk file-lock handoff has a fix
   sitting disabled in their tree — our 2.2–2.9× speech lead will not last
   forever by itself). **BACKLOG.**
