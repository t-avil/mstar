# Thinker Speculative / MTP Decode — Design & Feasibility

Scope: lift **decode throughput** and cut **ITL** for the Qwen3-Omni *Thinker
text decode* (the `thinker_decode` graph walk). This is the S2T / I2T output
stream and the same token stream that conditions the Talker for S2S / I2S, so a
faster Thinker decode helps every output modality. Talker-side codec
speculation is **design + profiling-gate only** (see §6).

Branch: `exp/spec-decode-mtp` (based on `integration-mnew` = M*-new).
Env gate: `MSTAR_THINKER_SPEC` (default **OFF** — fully inert when unset).

---

## 1. Feasibility verdict

**Feasible and worthwhile for the Thinker text decode** — with one important
correction to the original premise:

- **Qwen3 ships MTP, but this checkpoint does not include MTP head weights.**
  Verified against the on-disk `Qwen/Qwen3-Omni-30B-A3B-Instruct`
  `model.safetensors.index.json` (28,010 tensors):
  - Zero keys match `nextn | mtp | eagle | draft`.
  - The only Thinker head is a single `thinker.lm_head.weight`.
  - `thinker_config.num_hidden_layers == 48`, no extra prediction layer;
    `tie_word_embeddings: false`.
  - sglang-omni's port agrees: *"Qwen3MoE all layers are sparse and have no
    nextn now."*
  - The `"MTP"`-named code in `components/talker.py` and the baselines
    (`small_to_mtp_projection`, `mtp_block`) is the Talker's **multi-codebook
    codec depth predictor**, a different mechanism — not text MTP.

- **M\* has no token-level speculative-decode / draft-model / MTP infra.** The
  many `speculat*` hits in `graph/`, `engine/kv_cache_engine.py`,
  `engine/cuda_graph_runner.py` are *pipeline / cross-node* speculation
  (running node N+1 before N's output is confirmed) and *latent multi-token
  capture* hooks — useful plumbing to reuse, but not draft+verify token
  speculation. There is no proposer, no verify, no acceptance loop.

**Consequence — drafting strategy:**

| Strategy        | Status        | Why |
|-----------------|---------------|-----|
| MTP self-spec   | not available | no nextn head weights in checkpoint |
| EAGLE / Medusa  | needs training| requires a draft head trained on Thinker hiddens; none exist |
| **N-gram / prompt-lookup** | **implemented, weight-free** | zero new params; runs on stock checkpoint today |

So the default and immediately-usable path is **weight-free prompt-lookup
(n-gram) speculation**, with an **EAGLE** path scaffolded and its weight
requirement marked for a future trained head.

---

## 2. Faithfulness (the core invariant)

Speculative decoding is only a real win if it is **output-preserving**:

- **Greedy (temp == 0):** verify accepts draft token `i` iff it equals
  `argmax(target_logits[i])`; first mismatch commits the target's argmax and
  stops. Committed sequence is **bit-exact** to greedy decoding. Enforced by
  `ParityGate.check_greedy` (raises `ParityViolation` on any drift).
- **Sampling (temp > 0):** standard speculative-sampling rejection rule
  (Leviathan 2023 / Chen 2023) — faithful to the target distribution *in
  expectation*, **not** bit-exact step-for-step (RNG consumption differs).
  Parity is checked distributionally (acceptance rate / KL sanity), not by
  token equality. Implemented in `verify_sampling`.

This is why the design centers the verify + parity gate: the drafter can be
anything (n-gram, EAGLE, MTP) and correctness is unaffected — wrong guesses are
simply rejected.

---

## 3. Where it hooks into the decode loop

- Graph: `qwen3_omni_model.py` builds `thinker_decode` as a `Loop` whose
  `Thinker` node emits `new_token` (to client) and feeds `text_inputs` back to
  itself — one token per iteration (`get_graphs`, ~L556).
- Forward: `submodules.py :: ThinkerSubmodule.forward`, `thinker_decode` branch
  computes `logits = lm_head(hidden[-1:])` (~L869). **This is the single-token
  baseline and the hook point.** An inert, env-gated notice is wired here today
  (no behavior change).
- Sampling: `utils/sampling.py :: Sampler.sample` (deterministic FlashInfer;
  greedy = temperature 0 one-hot). The verify reuses this for the target dist.
- Token plumbing: `worker/worker.py` (`new_token_outputs` → `buffer_new_tokens`)
  and `conductor/conductor.py` consume one token per step; a multi-accept step
  must emit a **list** of committed tokens (already list-shaped at
  `buffer_new_tokens`).

**Reusable latent plumbing** (the GPU work item): `cuda_graph_runner.py`
comments already anticipate *"multi-token-per-request decode/spec capture"* and
tree-spec; `kv_cache_engine.py` has speculation rollback; `graph/graph_io.py`
has speculative buffers. The verify forward = a packed multi-token
`thinker_decode` over the `k` drafted tokens producing `k+1` logits; accepted
positions keep their KV, rejected positions roll back.

---

## 4. What's implemented vs stubbed

**Implemented (CPU, GPU-free, unit-tested):**
- `thinker_spec.py`:
  - `ThinkerSpecConfig.from_env` — env-driven config (`MSTAR_THINKER_SPEC*`).
  - `NgramProposer` — weight-free prompt-lookup drafter (runnable now).
  - `verify_greedy` — faithful, bit-exact greedy accept/reject + bonus token.
  - `verify_sampling` — distribution-faithful speculative rejection sampling.
  - `ParityGate` — raises on greedy divergence; tracks acceptance rate.
  - `ThinkerSpecController` — propose → verify → commit orchestration
    (model ops injected as callables, so it's testable without a GPU).
- `talker_profile.py` — Talker-vs-Code2Wav split profiler (gated, see §6).
- `test/modular/test_qwen3_omni_thinker_spec.py` — CPU tests for proposer,
  greedy verify bit-exactness, parity gate, controller.
- Inert env-gated hook in `ThinkerSubmodule.forward` (one-time notice only).

**Stubbed (needs GPU / weights):**
- `target_verify_fn`: the packed multi-token `thinker_decode` verify forward +
  KV extend/rollback wiring into the engine. (CPU logic is done; GPU plumbing
  is the work item.)
- Multi-token emit per decode step through worker/conductor (commit a list).
- `EagleDraftProposer.propose` + head loading — **requires a trained EAGLE
  head** (`MSTAR_THINKER_SPEC_EAGLE_PATH`); raises by design until provided.
- CUDA-graph capture for the multi-token verify shape (k+1 tokens/request).

---

## 5. Expected speedup

Faithful spec decode does **not** change quality; it changes throughput by
amortizing memory-bound decode steps over accepted tokens. Expected
**Thinker text-decode speedup ≈ 1.5–2.5×** wall-clock, with a proportional ITL
reduction, the usual faithful-spec range. Realized gain scales with mean
acceptance length `1 + num_accepted`:

- **N-gram / prompt-lookup:** large on repetitive / long-context / quoting text
  (code, JSON, retrieved/echoed spans) — acceptance can be high; ~no cost (and
  ~no gain) on novel prose. Best ROI for structured S2T/I2T outputs.
- **EAGLE (future, trained head):** more uniform ~1.8–2.5× across general text;
  needs the trained head this checkpoint lacks.

These are decode-phase figures; end-to-end S2S/I2S gains are diluted by the
prefill, Talker, and Code2Wav stages (Amdahl) — which is exactly why Talker
speculation must be profiled first (§6).

---

## 6. Talker-vs-Code2Wav profiling plan (gate before any Talker spec)

Rationale: VocalNet-class results show the vocoder (Code2Wav) can be ~70% of
RTF; MTP-style Talker speculation also changes the codec output distribution
(audio-quality risk). So **measure before optimizing**.

Plan (scaffolded in `talker_profile.py`, gated by `MSTAR_TALKER_PROFILE`):

1. Wrap `TalkerSubmodule.forward` (talker_decode: LLM + code-predictor depth
   loop) in `get_global_timer().time("talker")`.
2. Wrap `Code2WavSubmodule.forward` (code2wav_chunk: vocoder) in
   `time("code2wav")`. Use **CUDA events** (not host timers) since the two run
   on separate partitions/streams.
3. Run a representative S2S/I2S workload; `dump()` emits a JSON split.
4. **Decision rule:** `talker_fraction ≥ 0.5` → Talker dominates, cautious
   Talker speculation may be worth designing. Otherwise Code2Wav dominates →
   **do not** pursue Talker speculation; optimize the vocoder (batching, fp8,
   CUDA graphs, chunk size) instead.

No cross-time Talker speculation is implemented now (audio-quality risk).

---

## 7. GPU validation commands

```bash
# 0. CPU unit tests (no GPU): proposer / verify / parity / controller
pytest test/modular/test_qwen3_omni_thinker_spec.py -q

# 1. Baseline parity — spec OFF (default) must be unchanged
#    (run the existing S2T/I2T parity/answer test with MSTAR_THINKER_SPEC unset)
MSTAR_THINKER_SPEC=0 <existing S2T eval cmd>   # capture token stream A

# 2. Spec ON, weight-free n-gram, parity gate ON (fails loud on any drift)
MSTAR_THINKER_SPEC=1 \
MSTAR_THINKER_SPEC_METHOD=ngram \
MSTAR_THINKER_SPEC_K=4 \
MSTAR_THINKER_SPEC_NGRAM_N=3 \
MSTAR_THINKER_SPEC_PARITY=1 \
  <same S2T eval cmd>                          # token stream B
#    Greedy: stream B MUST equal stream A (bit-exact). Compare to confirm
#    faithfulness; record mean acceptance length + decode tok/s & ITL.

# 3. Talker-vs-Code2Wav split (go/no-go for Talker spec)
MSTAR_TALKER_PROFILE=1 \
MSTAR_TALKER_PROFILE_OUT=benchmark-personal/thinker-spec/talker_split.json \
  <existing S2S eval cmd>
#    Inspect talker_fraction / verdict in the JSON.
```

GPU device hygiene per workspace CLAUDE.md: pin `CUDA_VISIBLE_DEVICES` to the
fixed project devices, confirm idle via `nvidia-smi` before launch, wrap runs in
`timeout`, and clean up on every exit. None of the above was run here (no GPU
in this environment): this change is design + scaffold + `py_compile` only.
```
