# Parity mode (`MSTAR_PARITY_MODE`)

Parity mode is a single opt-in switch that makes the refactored Qwen3-Omni
engine (**M\*-new**) reproduce the previous engine (**M\*-old**, `main`)
**byte-for-byte on Speech-to-Speech (S2S)**. Its purpose is to *prove* the
refactor introduces zero correctness regression.

The performance optimizations stay **default-ON**. Parity mode is only flipped
on for a verification run; it is never the default.

## Enabling it

```bash
export MSTAR_PARITY_MODE=1
# optional: pin the fixed seed (default 1234)
export MSTAR_PARITY_SEED=1234
mstar serve qwen3_omni
```

On startup the engine logs a `MSTAR_PARITY_MODE active: ...` line listing every
value it forced.

## What it forces

M\*-new differs from M\*-old by exactly these *unconditional default changes*.
Parity mode reverts all of them, and nothing else:

| Setting | M\*-new default | M\*-old (and parity mode) |
| --- | --- | --- |
| `native_audio_encoder` | `True` (native) | `False` (HF wrapper) |
| `native_vision_encoder` | `True` (native) | `False` (HF wrapper) |
| `codec_chunk_frames` | `15` | `25` |
| `codec_left_context_frames` | `15` | `25` |
| sampling seed | per-request (md5 of request id) | fixed (`MSTAR_PARITY_SEED`) |

Sampling **temperatures are identical** new-vs-old (thinker 0.7, talker 0.9) and
are **not** changed. The prompt-layout flags are already env-gated default-OFF
and are left untouched.

Implementation:

- `mstar/utils/parity.py` — single source of truth (`parity_mode_enabled()`,
  `parity_seed()`).
- `mstar/model/qwen3_omni/config.py` (`Qwen3OmniModelConfig.__post_init__`) —
  reverts the encoder and Code2Wav defaults, and logs at startup. It sets the
  *base* defaults, so the existing per-flag overrides
  (`MSTAR_QWEN3_NATIVE_AUDIO_ENCODER` / `MSTAR_QWEN3_NATIVE_VISION_ENCODER`)
  still win — handy for re-enabling a single native encoder to bisect.
- `mstar/conductor/conductor.py` — when parity mode is on and the request does
  not pin its own `seed`, the conductor uses `parity_seed()` so sampling noise
  is deterministic. An explicit per-request `seed` always wins.

## Why not greedy decoding

The obvious recipe for two stacks to agree is greedy decoding (temperature 0 →
argmax → deterministic). **It cannot be used for the speech path:** a Thinker
temperature of 0 produces empty Talker embeddings, and the downstream
`torch.cat` over the (empty) Talker hidden states crashes.

So S2S byte-identity is achieved instead with a **fixed seed at the normal
sampling temperatures**. Both engines draw identical philox noise (same seed,
same per-step offset advance), so identical logits yield identical samples,
hence identical codec tokens, hence identical audio bytes.

## Verifying new == old, byte-for-byte

### 1. Self-consistency (single server, auto-runnable on GPU)

Necessary condition: a single parity-mode engine must be self-deterministic.

```bash
MSTAR_PARITY_MODE=1 MSTAR_PARITY_SEED=1234 mstar serve qwen3_omni  # http://localhost:8000
MSTAR_NEW_URL=http://localhost:8000/generate \
MSTAR_S2S_AUDIO_IN=/path/to/prompt.wav \
python -m pytest test/modular/test_qwen3_omni_s2s_byte_parity.py \
       -k self_consistency -s
```

`test_s2s_self_consistency_byte_identical` runs the same request twice and
asserts the audio bytes are identical (sha256). It auto-skips without CUDA or a
reachable server (so CI stays green).

### 2. Cross-engine identity (two servers, manual / CI step)

This is the actual `new == old` proof. It needs **both** servers up, so it is a
documented manual step, not an auto-run.

```bash
# terminal A — M*-old (main), in a separate worktree/checkout
MSTAR_OMNI_SEED=1234 mstar serve qwen3_omni            # e.g. http://localhost:8001
# old defaults already match parity (HF encoders, codec 25/25); no flags needed

# terminal B — M*-new with parity mode, SAME seed
MSTAR_PARITY_MODE=1 MSTAR_PARITY_SEED=1234 mstar serve qwen3_omni  # http://localhost:8000

# terminal C — compare bytes
MSTAR_NEW_URL=http://localhost:8000/generate \
MSTAR_OLD_URL=http://localhost:8001/generate \
MSTAR_S2S_AUDIO_IN=/path/to/prompt.wav \
python -m pytest test/modular/test_qwen3_omni_s2s_byte_parity.py \
       -k cross_engine -s
```

`test_cross_engine_byte_identity` posts the identical S2S request to both
servers and asserts the audio streams are byte-identical.

Pin **both** servers to the **same tokenizer snapshot**, the same audio input,
chat template, `max_output_tokens`, and seed. If bytes still differ, inspect the
first differing chunk: a divergence early in the stream is a real parity bug; a
single late codec-token flip is numerical noise to investigate, not necessarily
a regression.
