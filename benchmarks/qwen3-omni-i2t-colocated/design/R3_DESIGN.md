# R3: MSTAR_MULTISTEP_INGRAPH=N — in-graph zero-sync decode (blueprint)
GOAL: amortize the per-step host D->H floor N-fold -> close i2t B32 like-for-like tok/s -7% (decode-bound).
Today only the FORWARD is captured (cuda_graph_runner.py:673-676); sample/feed/advance/plan are post-replay host.
FEASIBLE with ONE real blocker: attention over GROWING length inside capture. Primary: reserve N KV pages up
front + in-graph seq_len counter (FlashInfer cudagraph mode, plan for base+N). Fallback: chain N single-step
subgraphs under one parent graph sharing the base+N wrapper. If pinned FlashInfer can't do in-buffer seq_len
-> fallback. THIS IS THE FEASIBILITY RISK.
PER-SLOT FIX (eiv2 bug): extend _intern_static_buffer key (config_idx,key)->(config_idx,key,slot_idx) so
next_token_ids/pos_3d/eos_len are slot-private (:456-491, slot_idx threaded :438-444). Assert distinct buffer ids.
IN-GRAPH: argmax token write (fused_temperature_softmax include_greedy sampling.py:101-154); GPU-resident feed
(DIRECT_FEED :2246); pos_3d +=1/step; per-slot eos_len int32 = min(eos_len, k+1) on first eos step (monotone,
capturable). ONE batched D->H of [bs,N] tokens + eos_len[bs] after N steps.
PARITY STOP (avoid async_sched +9% trap): overrun IN-GRAPH but TRIM on host BEFORE emit (worker.py:3873-3878);
consume eos_len SAME round-trip (not deferred). Emit tokens_i[:eos_len], DROP overrun. Never emit post-EOS.
ROLLOUT: Slice0 N=2 SHADOW-only (assert ingraph 2-step==2x single-step on B1, no emit change) -> Slice1 N=2
trim+emit B1->B32 A/B -> Slice2 N=4/8. Flag MSTAR_MULTISTEP_INGRAPH default 0. Require SAMPLE_RENDEZVOUS off.
CEILING: host floor amortizes N-fold. N=4 ~ -55-65% amortizable host component -> should close -7% w/ margin.
RISK: FlashInfer growing-len capture (primary blocker), slot-shared carry (eiv2), overrun-emit (async_sched),
KV underreserve. Each has a shadow/assert gate. MULTI-DAY, multi-turn CUDA surgery.
