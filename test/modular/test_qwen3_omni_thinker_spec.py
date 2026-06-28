"""CPU unit tests for the Thinker speculative-decode scaffold.

These exercise the GPU-free logic: the n-gram proposer, the faithful greedy
verify, the parity gate, and the controller orchestration. They do NOT require
a GPU or the model checkpoint. Run with: pytest test/modular/test_qwen3_omni_thinker_spec.py
(torch required at runtime; import is lazy so py_compile works without torch).
"""

import pytest

from mstar.model.qwen3_omni.thinker_spec import (
    NgramProposer,
    ParityGate,
    ParityViolation,
    ThinkerSpecConfig,
    ThinkerSpecController,
    VerifyResult,
    verify_greedy,
)

torch = pytest.importorskip("torch")


def _onehot_logits(token_ids, vocab=16):
    """Build logits whose argmax at row i is token_ids[i]."""
    rows = []
    for t in token_ids:
        v = torch.full((vocab,), -10.0)
        v[t] = 10.0
        rows.append(v)
    return torch.stack(rows, dim=0)


def test_ngram_proposer_finds_repeat():
    p = NgramProposer(n=2)
    # history "a b c a b" -> last 2 = [a,b], earlier [a,b] followed by c
    hist = [1, 2, 3, 1, 2]
    assert p.propose(hist, k=3) == [3]


def test_ngram_proposer_no_match_returns_empty():
    p = NgramProposer(n=3)
    assert p.propose([1, 2, 3, 4, 5], k=4) == []


def test_verify_greedy_all_accepted_adds_bonus():
    draft = [5, 6, 7]
    # target argmax matches all + one bonus row
    logits = _onehot_logits([5, 6, 7, 8])
    r = verify_greedy(draft, logits)
    assert r.accepted_tokens == [5, 6, 7, 8]
    assert r.num_accepted == 3
    assert r.acceptance_rate == 1.0


def test_verify_greedy_mismatch_emits_correction_and_stops():
    draft = [5, 6, 7]
    # target disagrees at position 1 (wants 99 not 6)
    logits = _onehot_logits([5, 99, 7, 8])
    r = verify_greedy(draft, logits)
    assert r.accepted_tokens == [5, 99]  # accept 5, correct to 99, stop
    assert r.num_accepted == 1


def test_verify_greedy_is_bitexact_to_greedy():
    # Faithfulness: committed tokens must equal plain argmax decode.
    draft = [1, 2, 3]
    target_choices = [1, 2, 9, 4]  # mismatch at idx 2
    logits = _onehot_logits(target_choices)
    r = verify_greedy(draft, logits)
    # plain greedy would have produced: 1, 2, then target's 9 (correction)
    assert r.accepted_tokens == [1, 2, 9]


def test_parity_gate_passes_on_match():
    g = ParityGate(enabled=True)
    g.check_greedy([1, 2, 3], [1, 2, 3])
    assert g.mismatches == 0


def test_parity_gate_raises_on_divergence():
    g = ParityGate(enabled=True)
    with pytest.raises(ParityViolation):
        g.check_greedy([1, 2, 4], [1, 2, 3])


def test_controller_step_faithful_against_baseline():
    cfg = ThinkerSpecConfig(enabled=True, num_speculative_tokens=3, method="ngram",
                            ngram_n=2)
    ctrl = ThinkerSpecController(cfg)
    history = [1, 2, 3, 1, 2]  # ngram will propose [3]

    # target_verify_fn returns logits whose argmax == greedy continuation.
    # Pretend the true greedy continuation of this history is [3, 7].
    def target_verify_fn(hist, draft):
        # rows = len(draft)+1; argmax row0 should agree with draft[0]=3,
        # bonus row = 7
        return _onehot_logits([3, 7][: len(draft) + 1])

    r = ctrl.step(history, target_verify_fn)
    assert isinstance(r, VerifyResult)
    assert history[-2:] == [3, 7]
    assert r.num_accepted == 1


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("MSTAR_THINKER_SPEC", "1")
    monkeypatch.setenv("MSTAR_THINKER_SPEC_K", "6")
    monkeypatch.setenv("MSTAR_THINKER_SPEC_METHOD", "ngram")
    cfg = ThinkerSpecConfig.from_env()
    assert cfg.enabled and cfg.num_speculative_tokens == 6
    assert cfg.parity_check  # defaults ON when enabled
