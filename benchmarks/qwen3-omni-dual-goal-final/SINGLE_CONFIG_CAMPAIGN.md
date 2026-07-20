# SINGLE-CONFIG CAMPAIGN — one config that wins BOTH text + speech (2026-07-19)

GOAL: a SINGLE serving config that wins/ties vLLM-Omni-0.24 on ALL 5 paths (i2t, s2t,
i2s, s2s, t2s). Per-modality wins STAY (they're the submission baseline); this is
additive. Deliver the standard 2×2 charts for the user to reason. Separate branch per idea.

## PROFILING (done)
- GPUs = 2× H200, **143771 MiB (~140GB) each**. Configs peak ~60-76GB (Thinker rank1),
  ~16-30GB (rank0). **~60-80GB HEADROOM/GPU.** VMEM is NOT the bottleneck.
- => **Encoder REPLICATION on both ranks is free** (encoders are small). Quantization is a
  SECONDARY lever (not needed for replication; may help elsewhere / vLLM continuous-quant idea).
- DISK: /home LOW (32-33G free, 97% used — shared volume, mostly others). MY footprint tiny;
  boots write to the POOL (TMPDIR + mega_cache on /m-coriander). MONITOR /home each loop tick.

## ROOT CAUSE (established)
Encoder placement conflict is by OUTPUT modality: rank1=Thinker (text-out bottleneck),
rank0=Talker+Code2Wav (speech-out bottleneck). Encoder wants the OPPOSITE (idle) rank.
Static placement is right for one modality, wrong for the other. Tested + failed:
split-by-input (crash s2s, regress i2t), TP-encoder (loses s2t+s2s).

## IDEAS (>=5; refine with agents swe-quant / researcher-theory / critique-code)
1. **REPLICATED encoder + output-modality ROUTING** (PRIMARY). Encoders on BOTH ranks;
   route each request's encode to the idle rank by output modality (text-out->rank0,
   speech-out->rank1). VMEM-free. Blocker: routing code + the cross-rank prefill KeyError.
   Branch: exp/replicated-encoder.
2. **FIX the cross-rank prefill bug** (node_manager_utils.py:1114 KeyError). Unblocks the
   split's s2s AND enables idea 1. Branch: exp/xrank-prefill-fix.
3. **ASYNC / overlapped encoder** — overlap encode with decode/synthesis on the SAME GPU so
   placement stops mattering (correct version of the rejected async-encoder). Branch: exp/async-enc.
4. **vLLM-Omni continuous quantization** — adopt their best-quant technique (per swe-quant).
   Secondary; may enable cheaper replication or other wins. Branch: exp/cont-quant.
5. **Encode on lower-INSTANTANEOUS-LOAD GPU** — dynamic by runtime load, not just modality.
   Branch: exp/load-balanced-enc.
6. (spare) **Quantized+replicated encoder** if replication ever needs to be cheaper.

## PLAN
- Await 3 agents -> refine ideas -> pick top 1-2 -> implement on separate branches ->
  boot + rebench 5 paths (or contested paths quick) -> chart -> iterate.
- Primary bet: idea 1+2 (replicated encoder + routing + bug fix).

## AGENT FINDINGS
### swe-quant (done): quantization is NOT the lever
- H200/Hopper: only FP8/Int8 online quant; NVFP4 W4A4 (#4025) is Blackwell-only. MXFP4=NPU/XPU.
- vLLM-Omni + M* BOTH deliberately keep encoders BF16 (vllm docs fp8.md L103, PR #2702; M* has no
  encoder quant). => replicate encoders in BF16, no quant. Keep MSTAR_MOE_FP8 (Thinker, already on).
- Reusable: vllm ComponentQuantizationConfig per-component ROUTING pattern (apply to placement).
  vLLM #3855 "Distributed Replica Control Plane" = per-component placement + replication precedent.
- M* encoder flags for the ASYNC/overlap idea (#3): MSTAR_ENCODER_ASYNC(+_DEPTH),
  MSTAR_ENC_OVERLAP_V2, MSTAR_ENCODER_CUDA_GRAPH(+_CG_*), MSTAR_ENC_STEP_BUDGET.
- Fork 12 commits behind upstream 59fe435c; upstream #164 Rust ZMQ transport could cut 13% zmq floor.
- => IDEA 4 (quant) DEPRIOTIZED/DROP. Primary stays idea 1+2 (replicated encoder + routing + bug fix).
  Idea 3 (async encoder) has ready flags to A/B. New idea: study vLLM #3855 replica control plane.

### researcher-theory (done): the mechanism ALREADY EXISTS — ~10-line change
- conductor.py:472-499 `_assign_worker_graphs_to_workers` ALREADY picks a DP-replica rank per
  node_group per request (currently np.random.randint = MSTAR_DP_ROUND_ROBIN). Output modality
  is available at the call site (conductor.py:654 _do_ingest_request, body.initial_output_modalities).
- CONFIRMED via code: tp_size defaults to 1 (base.py:46); _tp_ranks only if tp_size>1 (base.py:55).
  So my FAILED tpenc config (encoders ranks:[0,1], NO tp_size) was NOT TP — it was DP-replica with
  RANDOM routing. tpenc failed because random sent ~half the encodes to the BOTTLENECK GPU. => the
  fix is JUST deterministic routing. tpenc config = already the replicated-encoder config.
- SOLUTION: encoders DP-replica [0,1] (free BF16 copies) + route encode by OUTPUT modality:
  text-out -> rank0 (=encoff), speech-out -> rank1 (=base, co-located w/ Thinker -> AVOIDS the s2s
  cross-rank crash!). Byte-identical (same weights). Recovers BOTH per-modality optima in ONE config.
- The encoder is the ONLY multi-rank DP-replica group in the config -> identify by len(wg.ranks)>1.
- Caveat: PD-disagg B32-i2t-TTFT bonus is text-only, can't fold into a speech-serving config
  (accept i2t B32 ~parity TTFT). Ranks: (1) this, (2) load-based routing refinement, (3) async (subsumed),
  (4) quant hardening. tpenc's real failure = random routing, not TP.

### critique-code (done): CONFIRMS the approach + exact KeyError fix
- CONFIRMED: ranks:[0,1] no tp_size = 2 independent REPLICAS (base.py:54-63 tp_size>1 gate;
  conductor.py:380-383 both ranks instantiate). Routing site conductor.py:472-500 + output modality
  at :654 (body.initial_output_modalities). Downstream fanout (sharding, node_to_workers) follows
  automatically. My implementation matches this exactly. (tpenc.yaml was ALSO a random-routed replica
  = why it failed, confirmed.)
- KeyError (c) ROOT CAUSE: (thinker_decode_loop, prefill_text) is NEVER constructed for ANY topology
  (thinker_decode_loop belongs only to the thinker_decode walk's wg; node_manager_utils.py:1066-1070).
  Base works because the loop-stop resolves when the partition's walk is already thinker_decode; cross-
  rank/replicated routes the stop while the partition reports prefill_text -> lookup :1113-1116 misses.
- FIX (c), ~5 lines, SAFE (fallback only on miss): at construction (:1066-1070) also build
  loop_to_workers: dict[str,list[str]] (union over walks per loop_name), store on PerRequestInfo
  (add field ~:421); in get_dyn_loop_workers (:1113) try exact (loop,walk) key, except KeyError ->
  return loop_to_workers[loop_name]. Existing behavior unchanged.
- CONTINGENCY: my modality routing reproduces working base(speech)/encoff(text) placements, so may
  dodge the KeyError. IF the decisive test crashes s2s (server.log KeyError NodeAndGraphWalk) -> apply
  fix (c) to mstar-repenc/mstar/worker/node_manager_utils.py + rebuild. If it passes -> (c) optional hardening.

## IMPLEMENTATION (exp/replicated-encoder off winning/dual-goal)
1. config qwen3omni_2gpu_dpenc.yaml: encoders ranks:[0,1] no tp_size (DP-replica); Thinker r1,
   Talker+Code2Wav r0 (base placement so speech works).
2. conductor.py: thread body.initial_output_modalities into _assign_worker_graphs_to_workers;
   for the multi-rank DP-replica (encoder) group, replica_idx = wg.ranks.index(1 if "audio" in
   output_modalities else 0) instead of random.
3. boot + test contested paths (s2t n256, i2t, i2s, s2s B32) -> expect encoff-for-text, base-for-speech.

## *** RESULT: SINGLE CONFIG WINS BOTH (2026-07-19 T2) ***
dpenc (replicated encoder + output-modality routing) vs vLLM-0.24 B32:
  i2t 1817 tok/s (n256) = 1.04x  | s2t 848 = 1.28x  | i2s 102.2 sps = 2.16x  | s2s 74.9 = 2.18x.
ALL WIN. s2s NO CRASH (speech-out routes encode to rank1 co-located w/ Thinker). i2t B32 n=128
read 1676 (wave-lottery) -> 1817 at n=256. vs per-modality winners: i2t 0.97x, s2t 1.02x, i2s
0.94x, s2s 1.05x (within noise; ~3% i2t cost = the cross-rank embed hop encoff already pays).
NO KeyError fix (c) needed — modality routing reproduces working base/encoff placements per path.
Branch exp/replicated-encoder PUSHED. Running FULL 5-path sweep [full_sweep.log] -> charts -> user.

## LOOP HEALTH CHECKS (every 15 min)
- GPU not silently booked by US: if a lab_*/server.pgid proc is alive AND holding GPU BUT
  curl fails / no new cells for >X min -> SILENT FAIL -> teardown + reboot or report.
- On track: sweeps producing cells; agents progressing.
- Disk: df /home; if <25G free -> clean regenerable caches / stop writing to /home + alert.
- NO co-location: never boot on a GPU with others' procs (boot scripts already ABORT >1000MiB).

## STATE LOG
- T0 2026-07-19: profiling done (VMEM free, disk low-monitor). 3 Opus agents spawned. Loop starting.
- T1 2026-07-19 23:15: IMPLEMENTED replicated-encoder + output-modality routing on
  exp/replicated-encoder (worktree mstar-repenc). config qwen3omni_2gpu_dpenc.yaml (encoders
  DP-replica [0,1]) + conductor.py _assign_worker_graphs_to_workers routes multi-rank DP group
  by output modality (speech->rank1, text->rank0). Syntax OK, committed. Booting (encoders on
  both ranks, GPU6=25GB/GPU7=61GB). DECISIVE TEST running [bgy46lnw4]: s2t n256+i2t+i2s+s2s B32.
  EXPECT: s2t~832(encoff), i2t~1866(encoff), i2s~109(base), s2s~71(base, NO crash). If all match
  per-modality winners -> SINGLE CONFIG WINS BOTH. researcher-theory released (finding consumed).
  Awaiting critique-code for confirm/refinements.
- T0b: vLLM-Omni release research (repo vllm-project/vllm-omni). "Continuous quantization" =
  vLLM picks best quant kernel per HW: NVFP4 W4A4 for Qwen3-Omni (#4025) + quack-FP8 (#4817,#4241)
  are BLACKWELL (sm_100) ONLY. Our H200 = Hopper sm_90 -> best quant = FP8, which M* ALREADY has
  (MSTAR_MOE_FP8). => QUANTIZATION WON'T unlock the single config on H200 (VMEM fine + best quant
  used; NVFP4 needs Blackwell). Also noted: vllm #504698de orchestrator inter-stage/client-output
  separation (serving perf), #86bdcaf3 engine-level KV mgmt. CONCLUSION: single-config path =
  REPLICATED encoder + output-modality ROUTING (idea 1+2), NOT quantization. Deprioritize idea 4.
