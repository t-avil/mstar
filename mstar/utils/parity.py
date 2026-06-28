"""Single source of truth for ``MSTAR_PARITY_MODE``.

Parity mode is an opt-in switch that makes the refactored engine (M*-new)
reproduce the previous engine (M*-old) byte-for-byte on Speech-to-Speech, so
the refactor can be proven to introduce zero correctness regression. The
performance-oriented optimizations stay default-ON; parity mode reverts only
the handful of *unconditional default changes* that separate new from old:

  * native audio/vision encoders   -> off (fall back to the HF wrappers)
  * Code2Wav streaming chunk size  -> 25/25 (old) instead of 15/15 (new)
  * sampling seed                  -> a fixed, deterministic seed

It does NOT change sampling temperatures (identical new-vs-old) and does NOT
touch the env-gated prompt-layout flags (already default-off).

Read these env vars through this module only, so every consumer agrees on what
"parity mode" means.
"""
from __future__ import annotations

import os

# Default seed used when parity mode is on and the request does not pin its own
# ``seed``. Any fixed value works; both engines just have to agree. Override
# with ``MSTAR_PARITY_SEED`` if a different fixed seed is desired.
DEFAULT_PARITY_SEED = 1234

_TRUTHY = ("1", "true", "yes", "on")


def parity_mode_enabled() -> bool:
    """True iff ``MSTAR_PARITY_MODE`` is set to a truthy value."""
    return os.environ.get("MSTAR_PARITY_MODE", "").strip().lower() in _TRUTHY


def parity_seed() -> int:
    """The fixed seed to use under parity mode.

    Honors ``MSTAR_PARITY_SEED`` if set, else ``DEFAULT_PARITY_SEED``.
    """
    raw = os.environ.get("MSTAR_PARITY_SEED")
    if raw is None or raw.strip() == "":
        return DEFAULT_PARITY_SEED
    return int(raw)
