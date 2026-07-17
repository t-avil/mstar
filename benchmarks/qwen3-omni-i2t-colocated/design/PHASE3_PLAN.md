# Phase 3 Plan — decode-CPU-floor rewrite (colocated like-for-like)
## Researcher reframe (THE key insight)
Floor = host-resident AR control loop, NOT the GIL. token(N) must land on host to build input_embed(N+1)
+ advance MRoPE + decide stop -> per-step completion_event.synchronize (worker.py:3537) + D2H
_prematerialize_for_check_stop (3557). GPU 94% idle. GIL only prevents HIDING off-critical work.
Even infinite threads: token(N)->input(N+1) latency chain remains. R1 (exile emit ~22% + uuid4 10%) =
proximate ~30% floor. DEEPER = vLLM MRV2 puts the ENTIRE AR machine (sample/pos-advance/stop) on GPU.

## Ranked (order 2 -> 1 -> 3; only 1+2 reach MRV2 zero-sync ceiling)
R1 (do first, safe, keep momentum): uuid4->monotonic counter (MSTAR_FAST_TENSOR_UUID, tensors.py:502/734/826/975,
   ~10%, zero numeric risk canary) + exile emit tail (MSTAR_DECODE_EXILE_POSTPROC, ~22%). Parity-gate.
R2 (GPU-native STOP, enables R3): token==eos on GPU into persistent stop-mask, sync 1 pinned flag every K steps
   (stride). Removes per-step barrier(3537)+D2H(3557)=16%+8%. Overrun <=K tokens bounded (vLLM's tradeoff). Untried.
R3 (GPU-resident SELF-FEEDING decode = MRV2): in-graph argmax (submodules.py:1463 exists) + write sampled id to
   PER-SLOT PRIVATE next_token_ids/pos_3d buffer (precedent plan pos_ids cuda_graph_runner.py:439-443) + advance
   MRoPE GPU-side. Host never needs token(N). eiv2 DIED from shared-across-NUM_SLOTS=2 buffers (108-110/496) +
   rendezvous sync + baseline-35% — ALL fixable; salvage = per-slot private + NO syncfree. Never built on champion.
   Ceiling: decode GPU-bound -> closes i2t/s2t tok/s+ITL gap outright.
## TTFT COROLLARY (big): co-admission (MIXED_SPLIT_ATTN) washed because host-bound decode leaves NO GPU slack for
   prefill. Once R3 makes decode GPU-bound (~50% idle), slack reappears -> co-admission REVIVES -> B32 TTFT fix.
   So R3 is a PRECONDITION for the B32 TTFT win. Tested co-admission in the wrong regime before.
## Ceiling: >> 10 points if R2+R3 land (decode zero-sync + TTFT revival). Higher risk (CUDA graph/parity) -> gate hard.

## 5-agent synthesis (perf + skeptic + researcher)
PARITY PROTOCOL (skeptic, CRITICAL): greedy already nondeterministic run-to-run at concurrency>1 (dyn
batch x FP-nonassoc MoE). B32 bitwise-diff = INVALID test. GATES: B1 determinism N/N (clean) + B32
LENGTH-DIST match (~178 tok/req no drift) + emit-order/no-wedge. (=> split-attn 3/32 was normal, not a fail.)
RISK-ADJUSTED ORDER: 1) uuid4->counter (~10%, safest; ONLY tensors.py tensor_uuid 502/734/826/975 = internal
storage key, NOT GraphEdge/loop NAME uuid base.py:450/447 which routes stops — DO NOT touch those; namespace
counter per-worker+per-boot; gate scheduler ready-set order unchanged). 2) emit-exile (22% self-time BUT
GIL-valve law: EMIT_SIDECAR shipped only +3.1%; VERIFY BATCH_EMIT/EMIT_SIDECAR already ON in my boots -> if
so, send_pyobj is residual/conductor frames, little left). 3) on-device stop = R2 (ONLY if overrun TRIMMED
before client emit = same visible tokens; else +9% async_sched trap). 4) in-graph greedy = R3 (killed twice;
per-slot PRIVATE next_token_ids/pos_3d - VERIFY champion slotting first; NO syncfree; shadow-mode). DO NOT
co-implement R2+R3 blindly (= eiv2 composed collapse).
KEY REALISM: R1 may net only ~few % (GIL-valve). The REAL win = R3 (GPU-resident decode, removes token->host
->input dep) which also REVIVES B32 TTFT co-admission. Sequence: R1 (bank easy+validate parity gates) -> verify
emit flags -> R3 (per-slot, shadow-mode) as the ceiling play.

## CUDA + concurrency agents — REVISED priority (standout finding)
★ R1b STANDOUT: worker.py:3537 output.completion_event.synchronize() is REDUNDANT per-step barrier (the
researcher's AR-loop floor). check_stop copy self-gates (side.wait_event 4104), emit reuses it (3773),
_d2h self-gates (3929). DELETE 3537+3541, rely on per-copy event gates + SIDECAR_CHECKSTOP=1 polling
(4046-4069). PARITY-NEUTRAL numerically; correctness MEDIUM (audit no ungated host read of a GPU tensor
between 3537 and emit). Removes most of the 16% stall. HIGH value, LOW-MED risk. <-- best first real lever.
NOTE: sampled D2H is load-bearing for EMIT (one D2H/step, pinned, _d2h_stream 4130) -> on-device-stop is NOT
a sync lever (rejected). H->D feed ALREADY GPU-resident (MSTAR_DIRECT_FEED cuda_graph_runner.py:2246).
R3 (in-graph argmax, HIGH risk): fold sampler.sample+clone into replay; per-slot PRIVATE via extend
_intern_static_buffer key (config_idx,key)->(config_idx,SLOT_IDX,key) @456/475; greedy-only NO syncfree;
clone(2239) folds in (per-slot ring). Gate 16/16 B1 + shadow.
R1 EXILE design (concurrency agent): split _postprocess_batch @3672; tail(3673-3883) host-only owns/copies;
per-rid _tail_futures + extend held-guard @4257 (REMOVE waits tail); _drain_tail before non-spec get_next_batch
@4966; FIFO preserves order; SHADOW mode (MSTAR_DECODE_EXILE_POSTPROC_SHADOW) asserts inline==deferred emit stream.
ORDER: R1a uuid4 (bank+validate gates) -> R1b remove-3537-sync (best clean win) -> R1 exile-emit -> R3 (last, shadow).
