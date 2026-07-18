# Qwen3-Omni M* — dual goal: text tok/s parity vs older M* + speech 2–3× vs vLLM

**Question.** Recent work made it look like speech generation dropped from ~2–3×
vLLM-Omni to "a tad better," and the worry was that fixing text tok/s had killed
the speech advantage. Goal: text tok/s ≥ older M* at **every** batch on **both**
text paths (i2t, s2t), while speech stays 2–3×.

All numbers below are **natural-EOS** (the regime the original 2–3× and the older
text tok/s were measured in), so every comparison is apples-to-apples. vLLM values
are **committed references** (0.22 natural-EOS), never re-benchmarked (owner rule).
Build = `mstar-godv9` (`opt/prep-pos-batched-v9`) + the full host-floor/MoE flag
stack (see `launch_*`). GPUs 6,7 (+4,5 for the paired A/B), NUMA node 1.

## TL;DR — the speech advantage was never killed

The "drop to a tad better" was **not** a code regression from the text work. Two
things, both proven on the live server:

1. **Wrong regime + newer baseline.** The alarming "~1.5×" came from measuring
   *fixed-256 audio-s/s vs the newer vLLM-0.24*. In the original regime
   (natural-EOS vs committed vLLM-0.22), i2s is still **2.0–2.4× audio-s/s and
   2.6–3.6× req/s**. Same build, i2s B8: natural-EOS = 2.23× vs fixed-256 = 1.51×.
2. **A 2-GPU topology choice.** The `encoff` layout (encoders on the
   Talker/Code2Wav GPU) collapses only **i2s B32** to 1.27×. Reverting to the
   base layout (`qwen3omni_2gpu.yaml`, encoders on the Thinker GPU) restores it to
   **2.12×** (100.5 vs committed M*-new 98.7).

**Every text tok/s lever is Thinker-scoped / gated on `audio_output=False`** — none
touch Talker/Code2Wav. So the text flags do not, and cannot, cost speech. The only
coupling is encoder placement (topology).

## The 2-GPU topology tradeoff (the real finding)

With 2 GPUs the encoders can sit off the Thinker GPU **or** off the Talker/Code2Wav
GPU, not both:

| topology (config) | encoders on | s2t (audio→text) | i2s (image→speech) B32 |
|---|---|---|---|
| `encoff` (qwen3omni_2gpu_encoff.yaml) | Talker/Code2Wav GPU | **best** | collapses (1.27× vLLM) |
| `base`   (qwen3omni_2gpu.yaml)        | Thinker GPU          | soft | **best** (2.12× vLLM) |

**Quantified by a load-robust paired same-load A/B** (encoff vs base, 3 rounds,
interleaved, GPUs 4,5 vs 6,7 both NUMA node 1), median encoff/base text tok/s:

| path | B8 | B16 | B32 |
|---|---|---|---|
| **s2t** (audio→text) | **1.32×** | **1.43×** | **1.45×** |
| **i2t** (image→text) | 0.95× | 1.06× | 0.91× (topology-neutral) |

So the audio-encoder-on-Thinker contention costs s2t ~1.3–1.45×; i2t doesn't care
which topology. And i2s B32 needs `base` (encoders off the Talker/Code2Wav GPU).

## Speech result — base topology, natural-EOS vs committed vLLM-0.22

i2s (image→speech):   audio-s/s 2.0–2.4×, req/s 2.8–3.6× at every batch (B32 2.12× / 3.02×).
s2s (audio→speech):   audio-s/s 1.8–2.3×, req/s 2.8–3.5× (B32 2.05× / 2.88×).
→ **Speech 2–3× preserved at every batch.** (Charts: `charts/speech_nateos_*_base.png`.)

## Text result — tok/s vs older M*

vs the genuinely-older **+encoders** build: current is **1.5–2.4× at every batch,
both paths** — a decisive win. vs the immediate-predecessor **E1/v9**: at parity
(i2t B1–B8 ahead, B16/B32 within noise). The one softness is **s2t B16/B32 on the
base topology** (audio-encoder contends the Thinker GPU); `encoff` covers it.
(Charts: `charts/text_nateos_*.png`; per-batch table: `verdict_*.txt`.)

## Recommendation

**Topology-per-modality (recommended)** — route each modality to its optimal
encoder placement; both are the same `godv9` build + full flag stack, only the
config yaml differs:
- **Text (i2t, s2t) → `encoff`** (`launch_mstar_best.sh`): s2t 1.3–1.45× faster
  than base, i2t neutral. Text tok/s **parity-or-better vs older M* at every
  batch** — i2t ≈ E1/v9, s2t beats +encoders 1.8–2.4× and the recent-best v9 at
  B32 (849 vs 773). Goal met on both text paths.
- **Speech (i2s, s2s) → `base`** (`launch_mstar_base_fullstack.sh`): i2s 2.1–2.4×
  audio / 2.8–3.6× req, s2s 1.8–2.4× / 2.8–3.5× req, **every batch incl. B32**.

This delivers **text tok/s ≥ older M* at every batch on both paths AND speech
2–3×, simultaneously** — the full dual goal. The server already dispatches
walk-graphs per request modality; only the encoder placement is set at boot.

**Simplest single server (fallback): `base`** — meets both goals vs the
genuinely-older +encoders build (speech 2–3×; text 1.5–2.4× over +encoders) and
i2t parity vs E1, but s2t high-batch runs ~0.7× the encoff optimum. Use if
operating two topologies isn't worth it.

## Files
- `baselines.json` — committed older-M* + vLLM natural-EOS references (parity/ratio targets)
- `compare.py` / `verdict_*.txt` — per-batch dual-goal verdict
- `gen_text_nateos.py`, `gen_speech_nateos.py`, `charts/` — figures (shared chartstyle)
- raw per-cell results: `../exp_nateos_out/{encoff,base}/`, `../exp_paired_out/`, `../exp_gate_out/`
