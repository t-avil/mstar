"""Byte-for-byte S2S parity for Qwen3-Omni under ``MSTAR_PARITY_MODE``.

GOAL
----
Prove the M*-new refactor introduces zero correctness regression on
Speech-to-Speech (S2S) by showing it can reproduce M*-old's audio output
*byte-for-byte*. The performance optimizations stay default-ON; parity mode is
the opt-in switch that reverts the unconditional default changes M*-new made:

  * native audio/vision encoders -> off (HF wrappers, as in old)
  * Code2Wav streaming chunk      -> 25/25 (old) instead of 15/15 (new)
  * sampling seed                 -> fixed/deterministic

See ``mstar/utils/parity.py`` and ``docs/PARITY_MODE.md``.

WHY NOT GREEDY
--------------
The obvious way to make two stacks agree is greedy decoding (temperature=0 ->
argmax -> deterministic). That does NOT work for the speech path: a Thinker
temperature of 0 yields empty Talker embeddings, and the downstream
``torch.cat`` of the (empty) Talker hidden states crashes. So S2S byte-identity
must instead come from a FIXED SEED at the model's normal sampling temperatures
(thinker 0.7 / talker 0.9, identical new-vs-old). Parity mode pins that seed.

WHAT THIS MODULE AUTO-RUNS (GPU phase)
--------------------------------------
The deterministic *self-consistency* check: with parity mode on and a fixed
seed, the SAME engine must produce IDENTICAL audio bytes across two runs of the
SAME request. This is the runnable, single-server half of parity. It requires a
running M*-new server started with ``MSTAR_PARITY_MODE=1`` and CUDA; it
auto-skips otherwise (no GPU in CI, no server reachable).

CROSS-ENGINE BYTE-IDENTITY (manual / CI step, NOT auto-run)
-----------------------------------------------------------
The other half -- new == old -- needs BOTH servers up at once, so it is a
documented manual step rather than an auto-run:

  1. Check out M*-old (main) in a second worktree and start its server::

       MSTAR_OMNI_SEED=1234 mstar serve qwen3_omni            # old engine
       # -> note its URL, e.g. http://localhost:8001/generate

  2. Start M*-new with parity mode on, SAME seed::

       MSTAR_PARITY_MODE=1 MSTAR_PARITY_SEED=1234 mstar serve qwen3_omni
       # -> http://localhost:8000/generate

     Parity mode forces native encoders off + codec 25/25, so the ONLY
     remaining differences vs old are intended refactors that must not change
     output. Old needs no flags: its defaults already match (HF encoders,
     codec 25/25).

  3. Pin BOTH to the SAME tokenizer snapshot and the SAME request: same audio
     input file, same chat template, same max tokens, same seed (1234).

  4. Capture both audio streams and compare bytes::

       MSTAR_NEW_URL=http://localhost:8000/generate \
       MSTAR_OLD_URL=http://localhost:8001/generate \
       MSTAR_S2S_AUDIO_IN=/path/to/prompt.wav \
       python -m pytest test/modular/test_qwen3_omni_s2s_byte_parity.py \
              -k cross_engine -s

     ``test_cross_engine_byte_identity`` (below) implements this comparison but
     skips unless BOTH ``MSTAR_NEW_URL`` and ``MSTAR_OLD_URL`` are set, because
     it cannot bring up two servers itself.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os

import pytest

torch = pytest.importorskip("torch")

# --------------------------------------------------------------------------- #
# config (env-driven so the same module serves CI-skip, single-server GPU, and
# the manual cross-engine comparison)
# --------------------------------------------------------------------------- #
NEW_URL = os.environ.get("MSTAR_NEW_URL", os.environ.get(
    "MSTAR_OMNI_GENERATE_URL", "http://localhost:8000/generate"))
OLD_URL = os.environ.get("MSTAR_OLD_URL")               # set only for cross-engine
AUDIO_IN = os.environ.get("MSTAR_S2S_AUDIO_IN")         # path to input speech (.wav)
TEXT_IN = os.environ.get("MSTAR_S2S_TEXT_IN", "Hello, how are you today?")
VOICE = os.environ.get("MSTAR_S2S_VOICE", "ethan")
SEED = int(os.environ.get("MSTAR_PARITY_SEED", "1234"))
MAX_TOKENS = int(os.environ.get("MSTAR_S2S_MAX_TOKENS", "128"))
TIMEOUT_S = int(os.environ.get("MSTAR_S2S_TIMEOUT_S", "120"))


def _collect_audio_bytes(url: str) -> bytes:
    """POST one S2S request and return the concatenated raw audio bytes.

    Mirrors ``test/qwen3-omni/tts_request.py``: streams the newline-delimited
    JSON response and concatenates base64-decoded ``audio`` chunks in arrival
    order. Speech-in if ``MSTAR_S2S_AUDIO_IN`` is set, else text-in (still
    exercises Talker + Code2Wav, the components parity mode reverts).
    """
    requests = pytest.importorskip("requests")

    data = {
        "output_modalities": "audio",
        # Pin the seed on the request too, so the result is reproducible even if
        # the server was not started with MSTAR_PARITY_SEED.
        "model_kwargs": json.dumps({"voice": VOICE, "seed": SEED,
                                    "max_output_tokens": MAX_TOKENS}),
    }
    files = None
    if AUDIO_IN:
        data["input_modalities"] = "audio"
        files = {"audio": open(AUDIO_IN, "rb")}  # noqa: SIM115 (closed below)
    else:
        data["text"] = TEXT_IN
        data["input_modalities"] = "text"

    chunks: list[bytes] = []
    try:
        with requests.post(url, data=data, files=files, stream=True,
                           timeout=TIMEOUT_S) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("modality") != "audio":
                    continue
                b64 = msg.get("data", "")
                if b64:
                    chunks.append(base64.b64decode(b64))
    finally:
        if files:
            files["audio"].close()
    return b"".join(chunks)


def _require_server(url: str) -> None:
    requests = pytest.importorskip("requests")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA: S2S generation needs a GPU server")
    try:
        # Cheap liveness probe; any response (even 404/405) means it's up.
        requests.get(url.rsplit("/", 1)[0] + "/", timeout=2)
    except Exception:
        pytest.skip(f"no M* server reachable at {url} "
                    "(start one with MSTAR_PARITY_MODE=1)")


# --------------------------------------------------------------------------- #
# AUTO-RUN: deterministic self-consistency (single server, parity mode)
# --------------------------------------------------------------------------- #
def test_s2s_self_consistency_byte_identical():
    """Same engine + parity mode + fixed seed => identical audio bytes twice.

    This is the runnable half of parity: it does not need M*-old, only a single
    M*-new server started with ``MSTAR_PARITY_MODE=1``. Determinism here is the
    necessary condition for cross-engine byte-identity; if a single engine is
    not even self-deterministic, new==old is impossible.
    """
    _require_server(NEW_URL)

    first = _collect_audio_bytes(NEW_URL)
    second = _collect_audio_bytes(NEW_URL)

    assert first, "no audio bytes received from the M* server"
    h1, h2 = hashlib.sha256(first).hexdigest(), hashlib.sha256(second).hexdigest()
    assert len(first) == len(second), (
        f"audio byte length differs across runs: {len(first)} != {len(second)} "
        "(non-deterministic; check that MSTAR_PARITY_MODE=1 and the seed is fixed)")
    assert first == second, (
        f"audio bytes differ across two identical runs (sha256 {h1} != {h2}); "
        "parity mode is not deterministic")


# --------------------------------------------------------------------------- #
# MANUAL / CI: cross-engine byte-identity (needs BOTH servers; not auto-run)
# --------------------------------------------------------------------------- #
def test_cross_engine_byte_identity():
    """M*-new (parity mode) == M*-old, byte-for-byte, on the SAME S2S request.

    Skips unless BOTH ``MSTAR_NEW_URL`` and ``MSTAR_OLD_URL`` are set, since this
    module cannot launch two servers. See the module docstring for the full
    two-server procedure.
    """
    if not OLD_URL:
        pytest.skip("cross-engine check is manual: set MSTAR_OLD_URL "
                    "(and MSTAR_NEW_URL) to two running servers")
    _require_server(NEW_URL)
    _require_server(OLD_URL)

    new_bytes = _collect_audio_bytes(NEW_URL)
    old_bytes = _collect_audio_bytes(OLD_URL)

    assert new_bytes and old_bytes, "one of the servers returned no audio"
    hn = hashlib.sha256(new_bytes).hexdigest()
    ho = hashlib.sha256(old_bytes).hexdigest()
    assert new_bytes == old_bytes, (
        f"M*-new (parity) and M*-old audio differ: {len(new_bytes)} vs "
        f"{len(old_bytes)} bytes, sha256 {hn} != {ho}. Parity is NOT achieved; "
        "investigate the first differing chunk.")
