# Single-config experiment — can ONE topology win text AND speech? (2026-07-19)

## Question
The dual-goal result uses topology-per-modality: encoff (encoders on rank0 /
Talker+Code2Wav GPU) for TEXT out, base (encoders on rank1 / Thinker GPU) for SPEECH
out. Can a single static config win/tie both?

## Experiment: split-encoder config (my idea)
Hypothesis: the conflict is per-INPUT-encoder, so split them — vision_encoder->rank1,
audio_encoder->rank0 (configs/qwen3omni_2gpu_split.yaml). Measured B32 vs the per-
modality winners:
  s2t : 834.5 tok/s = 1.00x winner (encoff)   -> WINS text (audio off Thinker) ✓
  i2s : 105.4 audio-sps = 0.97x winner (base) -> keeps speech (vision off Talker) ✓
  i2t : 1653.8 tok/s = 0.89x winner (encoff)  -> TEXT REGRESSION (vision on Thinker) ✗
  s2s : CRASHES (worker-graph gap, node_manager_utils.py:1114 KeyError
        NodeAndGraphWalk(thinker_decode_loop, prefill_text) — cross-rank audio prefill
        for speech output isn't provisioned; same class as the known SIDE_PREFILL break) ✗

## Conclusion: NO static single config wins both — the conflict is by OUTPUT modality
Both encoders want the GPU OPPOSITE the bottleneck. The bottleneck is set by OUTPUT:
  * TEXT out (i2t, s2t): Thinker (rank1) is the bottleneck -> encoders want rank0.
  * SPEECH out (i2s, s2s): Talker+Code2Wav (rank0) is the bottleneck -> encoders want rank1.
Any fixed placement is right for one output modality and wrong for the other. The
split-by-input-modality is the wrong axis (it won s2t but lost i2t and crashed s2s).
topology-per-modality is optimal precisely because it flips encoder placement with the
bottleneck. Verified numerically: encoff/base i2t ~topology-neutral only at low batch;
at B32 vision-on-Thinker costs ~11% (i2t 0.89x), matching the split.

## The real single-"config" answer: DYNAMIC placement by output modality
Route the encoder to the idle rank PER REQUEST at admission (text-out -> rank0;
speech-out -> rank1), optionally with the encoder REPLICATED on both ranks (no cross-
rank transfer). This is a runtime placement decision — exactly M*'s "flexible placement"
thesis (arxiv 2606.12688), done per-request instead of per-deployment. Code change, not
YAML. Would ALSO need the cross-rank/side prefill provisioning bug fixed (the s2s crash).

## Status
Per-modality wins STAND (the submission result). This is exploratory: the split config
is committed for reproducibility; dynamic output-modality placement is the clear next
experiment for a true one-deployment win.

## Second candidate tested: TP-encoder (tensor-parallel encoders across both ranks)
configs/qwen3omni_2gpu_tpenc.yaml — encoders ranks:[0,1] so each modality pays HALF the
encoder contention. Boots (encoders ARE TP-shardable). B32: s2t=622.9 (0.94x vLLM024 =
LOSES), s2s=21.2 sps (0.62x vLLM024 = collapses), i2t/i2s crashed. FAIL: TP comm overhead
+ half-contention on BOTH bottleneck GPUs makes everything worse, not a compromise.

## FINAL: static single-config space exhausted — both candidates fail
  encoff (both->rank0): wins text, loses speech.        base (both->rank1): wins speech, loses text.
  split (audio->0,vision->1): wins s2t, regress i2t, CRASH s2s.   TP-enc: loses s2t + s2s.
No static placement wins both. The conflict is fundamentally by OUTPUT modality (bottleneck
= Thinker for text-out, Talker+Code2Wav for speech-out). ONLY dynamic per-request encoder
placement by output modality can win both — a runtime routing change (M*'s flexible-placement
thesis, per-request) + the cross-rank prefill provisioning fix. Per-modality topology stays
the submission result; dynamic placement is the clearly-scoped next experiment.
