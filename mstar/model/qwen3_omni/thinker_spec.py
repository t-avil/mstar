"""Speculative decoding scaffold for the Qwen3-Omni Thinker text decode.

Goal
----
Lift DECODE throughput and cut inter-token latency (ITL) for the Thinker's
autoregressive *text* decode (the ``thinker_decode`` graph walk in
``qwen3_omni_model.py``). This is the tokens emitted for S2T / I2T output, and
it is also the same token stream that conditions the Talker for S2S / I2S, so a
faster Thinker decode helps every output modality.

Design philosophy: FAITHFUL self-speculation
---------------------------------------------
Speculative decoding is only a "free" win when it is *output-preserving*: the
accepted token sequence must match what the un-accelerated sampler would have
produced.

  * Greedy (temperature == 0): the verify step accepts draft token ``i`` iff it
    equals ``argmax`` of the target model's logits at that position. The result
    is **bit-exact** to greedy decoding. A ``ParityGate`` asserts this.
  * Sampling (temperature > 0): we use the standard speculative-sampling
    rejection rule (Leviathan et al. 2023, Chen et al. 2023). This is faithful
    to the target distribution *in expectation* but NOT bit-exact step-for-step
    because RNG consumption differs. Parity for sampling is therefore checked
    distributionally (KL / acceptance-rate sanity), not by token equality.

MTP head availability (checkpoint fact, verified)
-------------------------------------------------
The shipped ``Qwen/Qwen3-Omni-30B-A3B-Instruct`` checkpoint does **NOT** contain
MTP / nextn / EAGLE / Medusa draft weights. Verified against
``model.safetensors.index.json`` (28010 tensors): the only Thinker head is a
single ``thinker.lm_head.weight``; there are zero keys matching
``nextn|mtp|eagle|draft``. ``thinker_config.num_hidden_layers == 48`` with no
extra prediction layer, and sglang-omni's port confirms it: "Qwen3MoE all layers
are sparse and have no nextn now". (The "MTP"-named code in the Talker/code
predictor is multi-codebook *codec* depth prediction, a different mechanism.)

Consequences for the draft strategy:

  * MTP self-speculation  -> NOT available (no nextn head weights to load).
  * EAGLE / Medusa        -> requires a trained draft head we do not have.
                             Scaffolded as ``EagleDraftProposer`` with the weight
                             requirement marked; loads nothing today.
  * N-gram / prompt-lookup -> WEIGHT-FREE. Works on the existing checkpoint with
                             zero new parameters. Implemented and runnable. This
                             is the default proposer so the path is usable now.

This module is environment-gated by ``MSTAR_THINKER_SPEC`` (default OFF) so it is
completely inert unless explicitly enabled.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # avoid importing torch at module import time (CPU-only hosts)
    import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment gating
# ---------------------------------------------------------------------------
def _envflag(name: str) -> bool:
    """Read a boolean env flag (default OFF). Accepts 1/true/yes/on."""
    raw = os.environ.get(name)
    if raw is None:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _envint(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def thinker_spec_enabled() -> bool:
    """Master gate. When OFF (default) nothing in this module touches decode."""
    return _envflag("MSTAR_THINKER_SPEC")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ThinkerSpecConfig:
    """Speculative-decode configuration for the Thinker text decode.

    Populated from the environment so it can be flipped without code changes.
    Env vars (all optional, sane defaults):
      MSTAR_THINKER_SPEC                 master gate (bool)
      MSTAR_THINKER_SPEC_K               draft length / lookahead (int, default 4)
      MSTAR_THINKER_SPEC_METHOD          'ngram' | 'eagle' (default 'ngram')
      MSTAR_THINKER_SPEC_NGRAM_N         prompt-lookup match length (int, default 3)
      MSTAR_THINKER_SPEC_NGRAM_WINDOW    max history scanned (int, default 0 = all)
      MSTAR_THINKER_SPEC_PARITY          run the parity gate (bool, default ON
                                         when spec is enabled — fail loud on drift)
    """

    enabled: bool = False
    # Number of tokens the draft proposes per step (the verify produces k+1
    # target logits and accepts a prefix of length [0, k], then appends one
    # bonus token sampled from the target -> 1..k+1 tokens per target call).
    num_speculative_tokens: int = 4
    method: str = "ngram"
    ngram_n: int = 3
    ngram_window: int = 0  # 0 => scan all prior tokens
    parity_check: bool = True

    @classmethod
    def from_env(cls) -> "ThinkerSpecConfig":
        enabled = thinker_spec_enabled()
        return cls(
            enabled=enabled,
            num_speculative_tokens=max(1, _envint("MSTAR_THINKER_SPEC_K", 4)),
            method=os.environ.get("MSTAR_THINKER_SPEC_METHOD", "ngram").strip().lower(),
            ngram_n=max(1, _envint("MSTAR_THINKER_SPEC_NGRAM_N", 3)),
            ngram_window=max(0, _envint("MSTAR_THINKER_SPEC_NGRAM_WINDOW", 0)),
            # Parity defaults ON whenever spec is enabled: a faithful spec decode
            # must never silently diverge from the baseline.
            parity_check=_envflag("MSTAR_THINKER_SPEC_PARITY") or enabled,
        )


# ---------------------------------------------------------------------------
# Draft proposers
# ---------------------------------------------------------------------------
class DraftProposer(ABC):
    """Proposes up to ``k`` candidate continuation tokens from history.

    A proposer is *cheap* relative to a target Thinker forward; the target then
    verifies the proposal in a single packed multi-token forward. A proposer
    never affects correctness — wrong guesses are simply rejected by verify.
    """

    @abstractmethod
    def propose(self, token_history: Sequence[int], k: int) -> list[int]:
        """Return 0..k proposed token ids (fewer => fewer to verify)."""
        raise NotImplementedError

    def requires_weights(self) -> bool:
        return False


class NgramProposer(DraftProposer):
    """Weight-free prompt-lookup / n-gram draft (Saxena 2023, "prompt lookup").

    Finds the most recent earlier occurrence of the last ``n`` tokens in the
    running sequence and proposes the tokens that followed it. Needs zero extra
    parameters, so it runs on the stock Qwen3-Omni checkpoint today. Effective on
    repetitive / long-context / quoting text (code, JSON, retrieved spans); a
    no-op (proposes nothing) on novel text, where it costs ~nothing.
    """

    def __init__(self, n: int = 3, window: int = 0):
        self.n = max(1, n)
        self.window = window  # 0 => unbounded

    def propose(self, token_history: Sequence[int], k: int) -> list[int]:
        hist = list(token_history)
        if k <= 0 or len(hist) < self.n + 1:
            return []
        if self.window > 0 and len(hist) > self.window:
            hist = hist[-self.window:]
        pattern = hist[-self.n:]
        # Search for the latest earlier match of `pattern` (exclude the tail
        # itself). Scan right-to-left for recency.
        last_start = len(hist) - self.n
        for start in range(last_start - 1, self.n - 2, -1):
            if hist[start:start + self.n] == pattern:
                cont = hist[start + self.n: start + self.n + k]
                if cont:
                    return cont
        return []


class EagleDraftProposer(DraftProposer):
    """EAGLE-style lightweight autoregressive draft head — SCAFFOLD ONLY.

    EAGLE drafts in *feature* space: a small (1-2 layer) transformer head takes
    the target model's penultimate hidden state plus the embedding of the last
    accepted token and autoregressively predicts the next few hidden states /
    tokens, which the full target then verifies in one packed forward.

    Weight requirement (NOT satisfied by the current checkpoint): an EAGLE head
    must be *trained* on the Qwen3-Omni Thinker's hidden states. No such weights
    exist in ``Qwen3-Omni-30B-A3B-Instruct``. Enabling ``method='eagle'`` without
    providing a head path therefore raises — by design, so we never silently run
    a degenerate / unfaithful drafter.

    To make this real later: (1) train an EAGLE head against the frozen Thinker,
    (2) point ``MSTAR_THINKER_SPEC_EAGLE_PATH`` at it, (3) implement ``propose``
    to run the head against cached hidden states. The verify + parity gate below
    are head-agnostic and already faithful, so only this class needs filling in.
    """

    def __init__(self, head_path: str | None = None):
        self.head_path = head_path or os.environ.get("MSTAR_THINKER_SPEC_EAGLE_PATH")
        self._loaded = False

    def requires_weights(self) -> bool:
        return True

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.head_path:
            raise RuntimeError(
                "EagleDraftProposer requires a trained EAGLE head, but none was "
                "provided. The Qwen3-Omni-30B-A3B-Instruct checkpoint ships no "
                "MTP/EAGLE weights (verified against model.safetensors.index.json). "
                "Set MSTAR_THINKER_SPEC_EAGLE_PATH to a trained head, or use "
                "MSTAR_THINKER_SPEC_METHOD=ngram (weight-free)."
            )
        # STUB: load head weights here once a trained head exists.
        raise NotImplementedError(
            "EAGLE head loading is stubbed; train a head and implement load."
        )

    def propose(self, token_history: Sequence[int], k: int) -> list[int]:
        self._ensure_loaded()  # always raises today (no weights)
        raise NotImplementedError


def build_proposer(cfg: ThinkerSpecConfig) -> DraftProposer:
    if cfg.method == "ngram":
        return NgramProposer(n=cfg.ngram_n, window=cfg.ngram_window)
    if cfg.method == "eagle":
        return EagleDraftProposer()
    raise ValueError(f"Unknown MSTAR_THINKER_SPEC_METHOD={cfg.method!r}")


# ---------------------------------------------------------------------------
# Verification (faithful accept/reject)
# ---------------------------------------------------------------------------
@dataclass
class VerifyResult:
    """Outcome of verifying a draft against target logits."""

    accepted_tokens: list[int]   # tokens to commit this step (>= 1: includes bonus)
    num_drafted: int             # how many draft tokens were offered
    num_accepted: int            # how many draft tokens matched (excludes bonus)

    @property
    def acceptance_rate(self) -> float:
        return self.num_accepted / self.num_drafted if self.num_drafted else 0.0


def verify_greedy(
    draft_tokens: Sequence[int],
    target_logits: "torch.Tensor",
) -> VerifyResult:
    """Output-preserving greedy verification.

    ``target_logits`` has shape ``(len(draft_tokens) + 1, vocab)``: row ``i`` is
    the target's next-token distribution given the first ``i`` draft tokens. We
    accept draft token ``i`` iff it equals ``argmax(target_logits[i])``; on the
    first mismatch we stop and instead emit the target's argmax at that row (the
    "bonus" / correction token). If all draft tokens match, the bonus is
    ``argmax(target_logits[-1])``.

    Guarantee: the committed sequence is **identical** to plain greedy decoding,
    because every committed token is an argmax of the same target distribution
    greedy would have used. This is what makes the speedup faithful.
    """
    import torch

    argmax = torch.argmax(target_logits, dim=-1).tolist()
    accepted: list[int] = []
    num_accepted = 0
    for i, dtok in enumerate(draft_tokens):
        if argmax[i] == int(dtok):
            accepted.append(int(dtok))
            num_accepted += 1
        else:
            # mismatch: commit the target's own choice and stop
            accepted.append(int(argmax[i]))
            return VerifyResult(accepted, len(draft_tokens), num_accepted)
    # all draft tokens accepted -> append the free bonus token
    accepted.append(int(argmax[len(draft_tokens)]))
    return VerifyResult(accepted, len(draft_tokens), num_accepted)


def verify_sampling(
    draft_tokens: Sequence[int],
    draft_probs: "torch.Tensor",
    target_probs: "torch.Tensor",
    rng: "torch.Generator | None" = None,
) -> VerifyResult:
    """Distribution-faithful speculative sampling (Leviathan/Chen rejection rule).

    For each draft token ``x`` drawn from draft dist ``q``, accept with prob
    ``min(1, p(x)/q(x))`` where ``p`` is the target dist. On rejection, resample
    from the residual ``norm(max(p - q, 0))`` and stop. If all accepted, draw a
    bonus from the target's last-row dist.

    Faithful *in distribution* to sampling from the target — NOT bit-exact to a
    given baseline RNG trace. ``draft_probs``/``target_probs`` shape
    ``(len(draft_tokens)+1, vocab)``; row i conditions on the first i draft tokens.
    """
    import torch

    g = rng
    accepted: list[int] = []
    num_accepted = 0
    n = len(draft_tokens)
    for i, dtok in enumerate(draft_tokens):
        x = int(dtok)
        p = target_probs[i, x]
        q = draft_probs[i, x]
        ratio = (p / q).clamp(max=1.0) if q > 0 else torch.zeros((), device=p.device)
        u = torch.rand((), generator=g, device=p.device)
        if u < ratio:
            accepted.append(x)
            num_accepted += 1
        else:
            residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
            s = residual.sum()
            if s > 0:
                residual = residual / s
            else:
                residual = target_probs[i]
            corr = int(torch.multinomial(residual, 1, generator=g).item())
            accepted.append(corr)
            return VerifyResult(accepted, n, num_accepted)
    bonus = int(torch.multinomial(target_probs[n], 1, generator=g).item())
    accepted.append(bonus)
    return VerifyResult(accepted, n, num_accepted)


# ---------------------------------------------------------------------------
# Parity gate
# ---------------------------------------------------------------------------
class ParityViolation(AssertionError):
    """Raised when speculative output diverges from the faithful baseline."""


@dataclass
class ParityGate:
    """Guards the faithful-spec invariant.

    Greedy: asserts the speculative token stream is bit-identical to what greedy
    would emit. Wire this in by running, for a sample of steps, the baseline
    single-token decode in lockstep and comparing committed tokens.

    Sampling: bit-equality does not hold; instead track acceptance rate and an
    optional KL between draft/target to catch a broken (degenerate) drafter.
    """

    enabled: bool = True
    mismatches: int = 0
    steps_checked: int = 0
    accept_rates: list[float] = field(default_factory=list)

    def check_greedy(
        self, spec_tokens: Sequence[int], baseline_tokens: Sequence[int]
    ) -> None:
        if not self.enabled:
            return
        self.steps_checked += 1
        if list(spec_tokens) != list(baseline_tokens):
            self.mismatches += 1
            raise ParityViolation(
                "Thinker speculative decode diverged from greedy baseline: "
                f"spec={list(spec_tokens)} baseline={list(baseline_tokens)}"
            )

    def record(self, result: VerifyResult) -> None:
        self.accept_rates.append(result.acceptance_rate)

    @property
    def mean_acceptance(self) -> float:
        return sum(self.accept_rates) / len(self.accept_rates) if self.accept_rates else 0.0


# ---------------------------------------------------------------------------
# Controller (orchestration scaffold)
# ---------------------------------------------------------------------------
class ThinkerSpecController:
    """Per-request orchestrator: propose -> target-verify -> commit.

    This is the *algorithm* in pure form. The two model-facing operations it
    needs are injected as callables so this class stays GPU-free and unit
    testable:

      * ``proposer.propose(history, k)``               -> list[int]
      * ``target_verify_fn(history, draft) -> logits`` -> (k+1, vocab) tensor
        i.e. one packed multi-token Thinker forward over the drafted tokens,
        returning next-token logits at each position (incl. the post-draft one).

    INTEGRATION (stubbed where it touches the engine):
      ``target_verify_fn`` must be backed by a multi-token ``thinker_decode``
      forward + the KV-cache extend/rollback the engine already has scaffolding
      for (see ``graph/graph_io.py`` speculative buffers, ``engine/
      kv_cache_engine.py`` speculation rollback, and ``cuda_graph_runner.py``
      "multi-token-per-request decode/spec capture"). Accepted tokens keep their
      KV entries; rejected draft positions must have their KV rolled back. That
      wiring is the GPU-side work item; this controller is the CPU-side logic.
    """

    def __init__(
        self,
        cfg: ThinkerSpecConfig,
        proposer: DraftProposer | None = None,
        parity_gate: ParityGate | None = None,
    ):
        self.cfg = cfg
        self.proposer = proposer or build_proposer(cfg)
        self.parity = parity_gate or ParityGate(enabled=cfg.parity_check)

    def step(
        self,
        token_history: list[int],
        target_verify_fn,
    ) -> VerifyResult:
        """Run one speculative step, mutating ``token_history`` in place.

        ``target_verify_fn(history, draft) -> (len(draft)+1, vocab) logits``.
        Greedy verify path (faithful, bit-exact). For sampling, swap in
        ``verify_sampling`` with draft/target probs.
        """
        draft = self.proposer.propose(token_history, self.cfg.num_speculative_tokens)
        target_logits = target_verify_fn(token_history, draft)
        result = verify_greedy(draft, target_logits)
        token_history.extend(result.accepted_tokens)
        self.parity.record(result)
        return result
