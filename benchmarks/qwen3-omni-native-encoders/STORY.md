# The Time‑to‑First‑Token / Inter‑Token‑Latency Trade‑off in Image‑to‑Text Serving
### Why matching vLLM‑Omni's responsiveness on M\* cannot be done without surrendering throughput

*An internal research report. Performance results are stated as ratios (how many times
faster or slower), not absolute numbers. No system internals are named. Companion
technical notes accompany this document for engineers.*

---

## Abstract

We set out to answer a recurring product question: can our in‑house image‑to‑text (I2T)
server, **M\***, be made to feel as responsive as **vLLM‑Omni** — that is, can it match
vLLM's time‑to‑first‑token (TTFT)? We approached it as an investigation rather than a
tuning exercise, because we already suspected the gap was structural. We (1) measured
the two systems head‑to‑head on the same hardware and inputs across a wide load range,
(2) reconstructed, from primary sources, *why* the two systems behave differently, and
(3) attempted to give M\* vLLM's responsiveness by adopting its scheduling strategy. The
three lines of evidence agree. vLLM achieves roughly **1.4× to 2.8× lower TTFT** (and,
critically, a TTFT that stays *flat* as load rises), while M\* delivers roughly **1.7× to
3.3× lower per‑token latency** (which stays flat while vLLM's degrades by about **5×**
under load) and roughly **2× higher throughput**. These are not independent dials: they
are three faces of one architectural choice. vLLM gets its TTFT by *mixing* new and
in‑progress work in every step; M\* gets its throughput and smoothness by *separating*
them so each step keeps a fixed shape that a recorded GPU routine can replay. When we
forced M\* to mix the way vLLM does, throughput collapsed by roughly **20× under load**
and requests began to fail — exactly the failure mode the architecture predicts, and the
same interference phenomenon the published literature reports (up to **28×** inflation of
per‑token latency for naïve mixing). The conclusion is that TTFT parity on I2T is a
deliberate, expensive architectural decision, not a missing optimization — and that the
responsiveness we *can* recover cheaply lies in the image‑ingestion pipeline, not in the
scheduler.

---

## 1. Motivation: an observation we did not think we could tune away

The investigation began with an observation. On image‑to‑text, our first‑token latency
trailed vLLM‑Omni's by a consistent margin, and — more tellingly — that margin **widened
as concurrency increased**. A latency gap that grows with load is the signature of a
*structural* difference, not a mis‑set parameter; a wrong knob produces a constant
offset, whereas an architectural mismatch compounds under pressure.

We also already knew two things about how the two systems are built, and those two facts
framed the whole study:

- **M\* is built on recorded, fixed‑shape GPU execution.** Its throughput advantage comes
  from replaying pre‑recorded sequences of GPU work ("CUDA graphs"), which only function
  when each unit of work has an identical shape every time.
- **vLLM, in its current generation, is built to mix freely.** It blends new requests'
  prompt‑processing into the same GPU step as the tokens it is already generating for
  other users, and it has a separate mechanism to keep that mixing from breaking its own
  recorded execution.

So we did not ask "what setting closes the gap?" We asked the architecturally honest
questions: *How large is the trade‑off, exactly? Why does it exist? And what would it
actually cost M\* to adopt vLLM's strategy?* This report answers all three.

---

## 2. Background: three metrics that pull against one another

Three numbers describe the user experience of a generative server, and they are in
tension:

- **Time‑to‑first‑token (TTFT)** — how long a user waits, after submitting, for the first
  piece of the answer. Governed by how quickly a new request is *picked up* and its
  prompt *processed* ("prefill").
- **Inter‑token latency (ITL)** — once generation starts, the gap between successive
  tokens. Governed by how cleanly the *generation* steps ("decode") run without being
  interrupted.
- **Throughput** — total requests served per unit time, which sets cost‑per‑request and
  capacity.

The tension is rooted in the fact that *prefill* and *decode* are fundamentally different
workloads. Prefill ingests a whole prompt at once — compute‑heavy, with a shape that
depends on prompt length. Decode emits one token per step — light, with a shape that
depends only on how many requests are running. A scheduler that favors getting prefills
done quickly improves TTFT but interrupts ongoing decodes (hurting ITL); a scheduler that
protects decodes improves ITL but makes new requests wait (hurting TTFT). No single
batch can be optimal for both, and the way a system resolves this conflict is the central
design decision behind everything that follows.

---

## 3. The throughput substrate: recorded execution and the tyranny of fixed shapes

The reason M\* is efficient is that it avoids re‑issuing the GPU's work instruction by
instruction on every step. Instead it *records* the entire sequence of GPU operations for
a step once and *replays* the recording thereafter, eliminating per‑instruction overhead.
On the small, frequent steps that dominate generation, this is worth on the order of a
third of the step's time — a large, compounding throughput advantage.

The technique carries one inviolable constraint, documented by the GPU vendor and the
underlying framework alike: **a recording is valid only for the exact tensor shapes it
was captured with.** Change the size of the work, and the recording no longer applies; the
system must fall back to issuing instructions one at a time — the slow path it was built
to avoid.

This constraint is benign for *decode* and hostile to *prefill*:

- In pure decode, every request contributes exactly one token, so the only thing that
  varies between steps is the *number of requests*. That is a small, enumerable set, so a
  system can record a modest library of routines — one per batch size — and round each
  real step up to the nearest. Decode is fully recordable.
- In prefill, and in any step that *mixes* prefill with decode, the per‑request token
  counts vary continuously with prompt length. The number of distinct shapes is
  effectively unbounded, so no finite library of recordings can cover them. Such a step
  cannot be replayed and must run on the slow path.

M\* resolves this by **never mixing**: it keeps prefill and decode in separate, fixed‑shape
steps so that every step stays on the fast recorded path. That choice is the source of
its throughput and ITL advantages — and, as we will see, the source of its TTFT
disadvantage.

---

## 4. Ahead‑of‑time compilation: necessary, but not the cure on its own

A natural question is whether ahead‑of‑time *compilation* of the model — tracing it once
and generating fused, optimized GPU code — sidesteps the shape problem. We examined this
directly, and the answer is nuanced.

Standard compilation specializes its generated code to a specific input shape. A new shape
forces a fresh, multi‑second recompilation in the middle of serving, producing latency
spikes. A "dynamic" compilation mode instead produces a single shape‑polymorphic artifact
that accepts any size — eliminating the recompilation spikes at the cost of slightly less
specialized code.

Our own measurements were deflationary about compilation as a *throughput* lever: for the
encoder‑style workloads we profiled, compiled execution did **not** beat ordinary eager
execution in steady state, because the work is dominated by large matrix multiplications
that the compiler cannot meaningfully accelerate. The value of dynamic compilation here is
therefore **tail‑latency hygiene** — removing recompilation stalls on novel shapes — not
average speed.

The deeper significance of dynamic compilation is that it is the *enabler* for the
strategy in the next two sections. A shape‑polymorphic compiled model is the building
block that lets a system record a *flexible* GPU routine — one that tolerates varying
token counts — which is precisely what is needed to record a *mixed* step. Compilation
alone is not the cure; compilation in service of flexible recording is.

---

## 5. Two scheduling philosophies

Faced with the prefill/decode tension of §2, the two systems made opposite choices.

**vLLM — mix everything (continuous batching).** vLLM's current engine generation abolishes
the separate prefill and decode phases and instead allocates a single per‑step token
*budget* across all active requests, giving each its decode token plus, optionally, a
chunk of some new request's prefill. New requests are absorbed into the very next step, so
they begin returning output almost immediately — excellent TTFT. The mechanism that bounds
the harm to ongoing decodes is **chunked prefill**: decodes are packed first, and only the
leftover budget is filled with prefill chunks, so a long prompt is sliced across several
steps rather than stalling generation in one. A single parameter — the per‑step token
budget — is the trade‑off dial: a larger budget pushes more prefill per step (better TTFT
and throughput, rougher ITL), a smaller budget protects the decodes (smoother ITL, slower
TTFT). Notably, this mixing behavior is a *recent* default in vLLM; its earlier generation
prioritized prefills and did **not** mix, which the project's own documentation describes
as optimizing TTFT at the expense of ITL and GPU efficiency — the very trade‑off space we
are mapping.

**M\* — keep clean, fixed‑shape lanes (separated/chunked steps).** M\* keeps new‑request
work and in‑progress work in separate, fixed‑shape steps so that every step remains a
recordable, replayable routine. A newcomer therefore waits a brief beat for the next
admissible slot rather than cutting straight into the current step. This preserves the
recorded fast path — and with it, throughput and ITL smoothness — at a small cost in
first‑token latency.

Read those two paragraphs together and the coupling is unavoidable. vLLM's responsiveness
*comes from* the mixing that breaks fixed‑shape recording; M\*'s efficiency *comes from*
the separation that costs responsiveness. They are not independent settings.

---

## 6. The architectural crux: how vLLM mixes without paying M\*'s penalty

If mixing breaks recorded execution, how does vLLM mix *and* stay fast? This is the single
most important finding of the study, because it explains why vLLM does not suffer the
collapse M\* would.

vLLM does not try to record a separate routine for every possible mixed shape. Instead it
**records around the one operation whose size genuinely varies** — the attention step. It
splits the model so that attention runs on the ordinary flexible path, while *everything
else* — the bulk of the per‑token arithmetic — is captured as a **single, shape‑flexible
recording** that accepts any number of tokens (built on the dynamic compilation of §4).
Because that flexible recording is valid for a prompt step, a generation step, or a mixed
step alike, vLLM keeps the fast recorded path **even while mixing**. It is, in effect, a
"mostly‑recorded" execution with a small flexible hole cut around the part that resists
recording. This is the current default behavior of its engine.

The contrast with M\* is now precise. M\* uses **whole‑step recordings tied to fixed
shapes** — a library approach. That library can cover decode (enumerable batch sizes) but
*cannot* cover an arbitrary mixed shape, so a mixed step in M\* has no valid recording and
falls to the slow path. This is not a quirk of M\*; it is the known limitation of
whole‑step recording, and other major serving stacks that started there have independently
migrated to the same "record‑around‑attention" strategy. In other words, the flexible‑hole
approach is the industry‑convergent answer to exactly the problem M\* faces, and M\* has not
adopted it.

The price vLLM pays for mixing is *not* throughput — it is ITL smoothness. Piling prompt
work into a generation step still makes that step heavier, so the users currently
receiving tokens get them a little less evenly. vLLM manages this with the token‑budget
dial and by serving in‑flight generations first, but the residual cost is real and shows
up clearly in our measurements (§8).

---

## 7. The interference is a known, quantified phenomenon

Our reasoning is corroborated by the peer‑reviewed literature, which studied this exact
trade‑off and named its failure mode. The key result is that prioritizing prompt
processing **interferes with ongoing generation**, producing "generation stalls" — pauses
in token output that, in a naïve mixing scheme, can stretch to seconds. Quantitatively,
the literature reports that combining a long prompt into a generation step can inflate the
time between tokens by **up to roughly 28×** relative to an uninterrupted generation step.
The proposed remedy is precisely chunked prefill with "stall‑free" batching — pack the
generations first, admit only as much prompt work as the leftover budget allows — which
bounds the interference, and which the throughput‑oriented systems subsequently adopted.
Under a fixed per‑token‑latency target, that discipline is reported to raise serving
capacity by roughly **2.6× to 5.6×** depending on model and hardware.

The significance for us is that the trade‑off we measured is not an artifact of our
harness or a property of one model. It is a documented, reproduced property of mixed
prefill/decode serving. The token budget is the dial; one end buys TTFT and throughput at
the cost of ITL, the other buys ITL at the cost of TTFT.

---

## 8. Empirical study: vLLM‑Omni vs M\* on image‑to‑text

We measured both systems on the same image‑to‑text workload and hardware, sweeping
concurrency from a single request up to heavy load. (We note that no official first‑token
or per‑token latency figures exist publicly for these omni‑class models on image input, so
this head‑to‑head is, to our knowledge, the actual data.) Expressed as ratios:

**Time‑to‑first‑token.** vLLM was faster at every load, by roughly **1.4× at light load
widening to about 2.8× at heavy load**. More importantly, vLLM's TTFT was **essentially
flat** across the entire load range, while M\*'s TTFT **grew by roughly 2.7×** from light to
heavy load. This is the continuous‑batching signature: a newcomer is absorbed immediately
regardless of how busy the system is.

**Inter‑token latency.** The ordering reverses. M\*'s per‑token latency was lower at every
load, by roughly **1.7× at light load widening to about 3.3× under load**, and it stayed
**flat**. vLLM's per‑token latency **degraded by roughly 5×** from light to heavy load — the
direct, measured manifestation of the generation‑stall interference of §7, as mixed prompt
work fattens the generation steps.

**Throughput.** M\* sustained roughly **2× the requests per second** of vLLM at every load
point — the dividend of never leaving the recorded fast path.

The picture is therefore coherent and complete. **vLLM is tuned to the TTFT corner of the
trade‑off** (mix aggressively, absorb newcomers instantly, accept rougher and
load‑sensitive per‑token latency and lower throughput). **M\* is tuned to the
throughput‑and‑smoothness corner** (separate the work, keep every step recordable, accept a
higher and load‑sensitive first‑token latency). Neither is "better"; they are two settings
of one dial, and our numbers are simply the two ends of it, measured.

---

## 9. The experiment: trying to give M\* vLLM's responsiveness

Because we saw the TTFT gap and knew that mixing is what closes it, we tried the obvious
thing: make M\* mix. The point was to test, empirically, whether M\* could move toward the
TTFT corner without paying for it. It cannot — and the way it fails is instructive.

**First, naïve mixing.** We allowed M\* to blend a waiting prompt into ongoing generation
steps, vLLM‑style, but on M\*'s fixed‑shape recording substrate. At light load nothing
changed, because mixed steps almost never arise when few requests overlap. Under heavy
load — the regime that matters — throughput **collapsed by roughly 20×** and the system
began dropping requests. The cause is exactly §3 and §6: every mixed step had an
off‑library shape, none could be replayed, and the system spent the load‑critical steps on
the slow path. This is the same interference the literature quantifies as a ~28× per‑token
penalty, here surfacing as an end‑to‑end throughput collapse.

**Second, trying to record the mixed shapes anyway.** We then attempted the only route
available to a fixed‑shape system: pre‑record routines for the mixed shapes too, across a
grid of "how many generations × how long a prompt chunk." We built and tested four
variants of this. The findings were uniformly cautionary. The space of mixed shapes is
large, so the recording library is large, which inflated memory use and pushed server
startup to many times its normal duration just to build the recordings. And the
implementation proved to be a correctness minefield: subtle defects that only manifest at
the high load where mixed steps actually fire, surfacing one after another as each was
fixed. We never obtained a clean high‑load result from the recorded‑mixed path; the effort
instead demonstrated *why* the fixed‑shape‑library route does not scale to the mixed‑shape
problem — which is the very reason the rest of the industry adopted the
record‑around‑attention approach instead.

The experiment thus closed the loop on the architecture: **M\* cannot cheaply mix, because
cheap mixing requires the flexible‑recording substrate (§6) that M\* does not have, and the
fixed‑shape‑library alternative is combinatorially and operationally prohibitive.**

---

## 10. Why image inputs make the trade‑off sharper still

Image‑to‑text is the *worst* case for the trade‑off, for reasons specific to images.

A text prompt has a single, scalar length. An image, by contrast, is expanded into a
**variable number of internal tokens that scales with its resolution** — from a few hundred
for a small image to several thousand for a large one, spanning more than an order of
magnitude. Each image thus contributes a *different* prompt shape, multiplying the very
shape‑variability that defeats fixed‑shape recording. Where a fixed‑shape system already
struggles to record mixed text steps, images make the space of shapes dramatically larger.

Image inputs also impose an intrinsic first‑token floor: the entire image must pass through
a vision stage before the first text token can be produced, and that vision stage runs on
the flexible (un‑recorded) path by default in the comparison system as well. Worse, a heavy
image prefill blocks concurrently generating requests — head‑of‑line blocking that inflates
*their* per‑token latency. This is precisely why the throughput‑oriented stacks added
encoder‑output caching and prompt chunking specifically for images.

The upshot is that the inputs where snappy first‑token latency is most desired are exactly
the inputs where matching it via mixing is most punishing. There is, however, a silver
lining: a meaningful share of image first‑token latency lives in the *ingestion* pipeline —
the vision stage and its hand‑off into generation — which can be streamlined **without**
disturbing the recording substrate, and therefore **without** the throughput penalty that
mixing incurs. This is the cheap responsiveness that remains on the table.

---

## 11. Discussion and recommendation

The three lines of evidence — measurement, architecture, and our own implementation
attempt — converge on one statement: **TTFT, ITL, and throughput on image‑to‑text are
three readings of a single dial whose position is set by the execution substrate.** vLLM
has set the dial toward first‑token responsiveness by mixing, and it pays for the privilege
in per‑token smoothness (which degrades under load) and in throughput; it can afford to mix
only because it records around the flexible part of the model. M\* has set the dial toward
throughput and smoothness by separating the work, and it pays in first‑token latency,
because its recordings demand fixed shapes.

To *match* vLLM's first‑token responsiveness, M\* would have to mix; to mix without the
throughput collapse we measured, M\* would have to replace its fixed‑shape‑library
recording with the flexible record‑around‑attention substrate — a substantial
architectural undertaking, not a configuration change. The honest menu is therefore:

1. **Match TTFT the cheap way (just mix as‑is): rejected.** It costs roughly an order of
   magnitude of throughput under load and drops requests. Not viable.
2. **Match TTFT the sound way (adopt flexible record‑around‑attention execution):**
   technically the right long‑term answer and the direction the industry has converged on,
   but a major investment that trades some per‑token smoothness for the ability to mix.
   Appropriate only if first‑token parity becomes a hard product requirement.
3. **Recover the TTFT that is genuinely free: recommended now.** Streamline the image
   ingestion and hand‑off, which lowers first‑token latency without touching the throughput
   machinery, narrowing the gap at zero throughput cost.

Our recommendation is to pursue (3) immediately and to treat (2) as a deliberate,
well‑scoped roadmap decision rather than a bug to be fixed — and, crucially, to stop
treating a TTFT *match* on image‑to‑text as a tuning oversight. It is not. It is a
considered position on a fundamental trade‑off, and this study is what lets us say so with
measurements and architecture rather than as an opinion.

---

## 12. Conclusion

We began from an observation — a first‑token gap on image‑to‑text that widened with load —
and a suspicion that it was structural. The suspicion was correct. Measured head‑to‑head,
vLLM leads on first‑token latency by roughly 1.4–2.8× and holds it flat under load, while
M\* leads on per‑token latency by roughly 1.7–3.3× (holding it flat while vLLM's worsens by
about 5×) and on throughput by roughly 2×. The cause is a single architectural fork:
vLLM mixes and records flexibly around attention; M\* separates and records whole fixed
shapes. When we forced M\* across the fork without changing its substrate, throughput fell
by roughly 20× under load — the predicted, and literature‑corroborated, price of mixing on
a fixed‑shape engine. First‑token parity on image‑to‑text is thus available to M\* only
through a substantial substrate change; what is available cheaply is the image‑pipeline
latency, and that is where we recommend spending effort now.

---

## References (primary sources consulted)

- vLLM, *V1 engine alpha* (unified token‑budget scheduler; piecewise CUDA graphs),
  blog.vllm.ai/2025/01/27/v1-alpha-release.html
- vLLM, *torch.compile integration* and *CUDA Graphs* design docs
  (record‑around‑attention / splitting around attention; capture modes),
  docs.vllm.ai — design/torch_compile, design/cuda_graphs
- vLLM, *Optimization & tuning* (per‑step token budget as the TTFT↔ITL dial; chunked
  prefill defaults), docs.vllm.ai — configuration/optimization
- vLLM, issue #4056 (decode‑only graphs; eager fallback for mixed batches in the earlier
  engine), github.com/vllm-project/vllm/issues/4056
- Agrawal et al., *Taming the Throughput‑Latency Trade‑off in LLM Inference with
  Sarathi‑Serve*, OSDI 2024, arXiv:2403.02310 (generation stalls; up to ~28× token‑latency
  inflation; chunked stall‑free batching; 2.6–5.6× capacity under SLO)
- PyTorch CUDA semantics and NVIDIA developer notes (static‑shape constraint on recorded
  GPU execution)
- Qwen2.5‑VL / Qwen2.5‑Omni / Qwen3‑Omni technical reports (Thinker–Talker structure;
  dynamic‑resolution vision tokenization; image‑token counts scaling with resolution),
  arXiv:2502.13923, arXiv:2503.20215, arXiv:2509.17765
- vLLM multimodal processing, prefix caching, and multimodal CUDA‑graph design docs
  (encoder caching; image prefill and head‑of‑line blocking), docs.vllm.ai — design/mm_processing,
  design/prefix_caching, design/cuda_graphs_multimodal
- Convergent record‑around‑attention in other stacks (SGLang piecewise CUDA graph;
  TensorRT‑LLM piecewise CUDA graph documentation)
