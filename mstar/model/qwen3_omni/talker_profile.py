"""Talker-vs-Code2Wav latency-split profiler (DESIGN + SCAFFOLD, gated).

Why this exists before any Talker speculation
---------------------------------------------
The literature (e.g. VocalNet) reports that for codec-LM TTS the *vocoder*
(Code2Wav here) can dominate the real-time factor — up to ~70%. By Amdahl's law,
speeding up the Talker (the codec LM) is pointless if Code2Wav is the bottleneck,
and MTP-style Talker speculation additionally changes the codec output
distribution, risking audible audio-quality regressions.

So the FIRST step for any Talker-side speculation is to *measure* the split.
This module times the two stages on the real pipeline and reports the fraction
of audio-path wallclock spent in each. Decision rule:

    talker_fraction >= TALKER_DOMINATES_THRESHOLD (default 0.5)
        => Talker dominates; cautious Talker speculation may be worth designing.
    otherwise
        => Code2Wav dominates; do NOT pursue Talker speculation. Optimize the
           vocoder (batching, fp8, CUDA graphs, chunk size) instead.

This module ONLY measures. It implements no Talker speculation. Gated by
``MSTAR_TALKER_PROFILE`` (default OFF), so it is inert in normal runs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field


def talker_profile_enabled() -> bool:
    raw = os.environ.get("MSTAR_TALKER_PROFILE")
    return raw is not None and raw.strip().lower() in ("1", "true", "yes", "on")


# Decision threshold: Talker must own at least this fraction of audio-path time
# before Talker speculation is even worth designing.
TALKER_DOMINATES_THRESHOLD = float(
    os.environ.get("MSTAR_TALKER_DOMINATES_THRESHOLD", "0.5")
)


@dataclass
class StageTimer:
    """Accumulates wallclock per pipeline stage.

    On GPU, wrap calls in ``torch.cuda.synchronize()`` (or use CUDA events)
    around the measured region so async kernel launches are attributed
    correctly — host-side ``time.perf_counter`` alone undercounts GPU work.
    The ``record`` API is sync-agnostic so the caller controls the barrier.
    """

    totals_ns: dict[str, int] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def record(self, stage: str, elapsed_ns: int) -> None:
        self.totals_ns[stage] = self.totals_ns.get(stage, 0) + elapsed_ns
        self.counts[stage] = self.counts.get(stage, 0) + 1

    def time(self, stage: str):
        """Context manager: ``with timer.time('talker'): ...``."""
        return _StageScope(self, stage)

    def summary(self) -> dict:
        talker = self.totals_ns.get("talker", 0)
        code2wav = self.totals_ns.get("code2wav", 0)
        denom = talker + code2wav
        talker_frac = talker / denom if denom else 0.0
        return {
            "talker_ns": talker,
            "code2wav_ns": code2wav,
            "talker_steps": self.counts.get("talker", 0),
            "code2wav_steps": self.counts.get("code2wav", 0),
            "talker_fraction": talker_frac,
            "code2wav_fraction": (1.0 - talker_frac) if denom else 0.0,
            "talker_dominates": talker_frac >= TALKER_DOMINATES_THRESHOLD,
            "threshold": TALKER_DOMINATES_THRESHOLD,
            "verdict": (
                "talker-dominates: cautious talker speculation may be worth it"
                if talker_frac >= TALKER_DOMINATES_THRESHOLD
                else "code2wav-dominates: do NOT pursue talker speculation"
            ),
        }

    def dump(self, path: str | None = None) -> dict:
        s = self.summary()
        out = path or os.environ.get("MSTAR_TALKER_PROFILE_OUT")
        if out:
            with open(out, "w") as f:
                json.dump(s, f, indent=2)
        return s


class _StageScope:
    def __init__(self, timer: StageTimer, stage: str):
        self.timer = timer
        self.stage = stage
        self._t0 = 0

    def __enter__(self):
        self._t0 = time.perf_counter_ns()
        return self

    def __exit__(self, *exc):
        self.timer.record(self.stage, time.perf_counter_ns() - self._t0)
        return False


# Process-global timer the submodules can reach without threading an object
# through the engine. Only used when MSTAR_TALKER_PROFILE is ON.
_GLOBAL_TIMER: StageTimer | None = None


def get_global_timer() -> StageTimer:
    global _GLOBAL_TIMER
    if _GLOBAL_TIMER is None:
        _GLOBAL_TIMER = StageTimer()
    return _GLOBAL_TIMER


# ---------------------------------------------------------------------------
# Integration plan (where to wire the two scopes — STUBBED, no behavior change)
# ---------------------------------------------------------------------------
# In submodules.py:
#   * TalkerSubmodule.forward (talker_decode walk): wrap the LLM forward +
#     code-predictor depth loop in `get_global_timer().time("talker")`.
#   * Code2WavSubmodule.forward (code2wav_chunk walk): wrap the vocoder forward
#     in `get_global_timer().time("code2wav")`.
# Because Talker and Code2Wav run on separate partitions/streams, prefer CUDA
# events over host timers for the GPU measurement, then aggregate per request.
# At request end, call get_global_timer().dump() to emit the split JSON, which
# feeds the go/no-go decision for Talker speculation.
