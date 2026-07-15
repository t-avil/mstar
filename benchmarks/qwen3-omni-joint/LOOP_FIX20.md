# LOOP_FIX20 — implement/bench/integrate the 20 TTFT fixes

Started 2026-07-12 17:15Z. 8h floor ends ~01:15Z (07-13). Wakeups every 10 min.
Source plan: LEARNINGS_TTFT.md (THE 20 FIXES). Research: RESEARCH_VLLM_OMNI_TTFT.md,
RESEARCH_MSTAR_TTFT_PATH.md. Code: worktree /m-coriander/coriander/tim/mstar-fix20
(branch opt/fix20 off opt/prep-pos-batched-v9 @ 1e7f10da). PYTHONPATH trap applies.

## DONE criteria (user)
i2t B32 on the fix20 stack: TTFT p50 better than current best (< 600ms quiet OR
< 1.0s at load ~25 where current gives 1.4-3.4s) AND tok/s raw >= 1536 (beats our
current best; vLLM band 1510) AND rps >= 8.0. s2t B32 no regression (>= 35 rps,
>= 664 tok/s). Scrappy cells first (n=32-64, paired A/B); winners re-benched
thoroughly (n>=96 i2t / n>=256 s2t) before integration.

## GPU rule
NEVER co-schedule. Box currently 100% held by root sglang+Megatron job (all 8 GPUs,
load 90-260). gpu_watch.sh polls for a free pair (prefer 6,7) -> fix20/gpu_watch.log.
No GPU work until PAIR_FREE. Implementation proceeds regardless (CPU-only).

## Triage of the 20 (from LEARNINGS_TTFT.md)
ALREADY VALIDATED (no work): #5 PD topology, #15 bench protocol, #20 keep mixed on.
CONFIG/BENCH-ONLY: #14 CPU-set pinning (launch script), #16 length-matched cells,
  #17 greedy default, #19 DP long-output cert. Need GPU only.
WAVE 1 (small/medium code, disjoint files, flag-gated default-off):
  #4 MSTAR_ADMIT_FASTPATH — arrival-triggered first-prefill admission bypassing
     spec-yield gate. Agent A.
  #2 MSTAR_ENC_STEP_BUDGET — per-step embed-token budget batching ALL pending
     image encodes into one varlen forward (vs grid-capped waves). Agent C.
  #8 MSTAR_EMIT_RID_INDEX + #9 MSTAR_EMIT_INLINE_FASTPATH + #18 MSTAR_ARGMAX_FAST
     — ordered-emit O(1) cleanup, empty-queue inline shortcut, temp==0 batched
     argmax. Agent B.
WAVE 2 (after wave-1 A/B): #6 producer-side seqnums (replaces consumer FIFO;
  supersedes #8/#9 if it wins), #7 plan reuse per vision wave, #10 shared
  sample+unpack, #11 batched WGD msgspec, #12 sidecar for i2t rids.
WAVE 3 (big): #1 one-step co-admission (arrival-time chunk fold, budget ~32k),
  #3 async scheduling (DEFER-SAMPLE narrow), #13 off-process detok.

## Bench plan per fix (when PAIR_FREE)
Boot fix20 server (ship flags + fix flag OFF) on pair; scrappy paired A/B:
flag OFF cell then ON cell back-to-back (i2t B32 n=48 w=3), same load window.
Win = TTFT or tok/s better outside noise with no correctness change (spot-check
outputs). Winner -> keep flag in ship set; loser -> document + park. Cumulative
stack re-bench thorough at end (n=96/256 + s2t guard cell).

## Status board
| fix | status | result |
|---|---|---|
| #4 admit fastpath | agent A implementing | - |
| #2 enc step budget | agent C implementing | - |
| #8+#9+#18 emit/argmax | agent B implementing | - |
| #14 cpu pinning | script pending (me) | - |
| #6 seqnums | wave 2 | - |
| #7 plan reuse | wave 2 | - |
| #10-#12 | wave 2 | - |
| #1 co-admission | wave 3 | - |
| #3 async sched | wave 3 | - |
| #13 detok proc | wave 3 | - |
| #16/#17/#19 | bench-only, need GPU | - |

## LOG
- 17:15Z loop start. All 8 GPUs held by root job (load 90-260). Worktree
  mstar-fix20 created @ 1e7f10da. gpu_watch launched. Wave-1 agents spawned.
- 19:28Z LOOP RESTART (user re-authorized; pair 6,7 free, only GPU0 busy by other
  user). New 8h floor ends ~03:30Z (07-13). Partial #4/#2 edits reverted; #8/#9/#18
  kept (data_worker.py, sampling.py). lab_server.sh gained LAB_CPUSET (fix #14,
  guarded physcpubind). fix20 server booting on 6,7:8345 (ship flags, new flags
  OFF, ttl 4h, setsid; boot.log). Agents fix4b-admit + fix2b-encbudget respawned
  (flags must be dynflag-toggleable). Load ~78 — TTFT cells need paired A/B.
  BENCH SEQ when READY: baseline OFF cell i2t B32 n=48 w=3 -> dynflags ON
  (EMIT_RID_INDEX+EMIT_INLINE_FASTPATH+ARGMAX_FAST) -> ON cell -> token-diff
  spot-check (argmax tie-break) -> repeat pair once. Then #14: reboot with
  LAB_CPUSET=64-87,192-215 paired A/B if load still high.
- 20:07Z workers died 20:02 = node1 RAM starvation (19.5G free of 774G; strict membind alloc fail; neighbors' 700G-RSS jobs + OOM-killer active 16-18h). Killed zombie lab. fix20_boot2.sh launched: waits node1>=120G + 6,7 free -> reboot -> exec bench1. Inductor cache partially warmed (faster next boot).
- 20:18Z boot3 launched (pid 2870618): RAM gate passed via box-avail 625G, LAB_MEMBIND_MODE=preferred (node1 43G). lab_server.sh now supports preferred membind. #7 NO-GO documented (plans per-request distinct; vectorized call already = BATCH_VISION_PREFILL; report fix7_plan_reuse.md). Footgun relearned x2: pgrep/ps -f self-match when target name in own cmdline — use [b]racket patterns + separate calls.
| w1a | off1 | 6.8s | 4.27 | tok=740.53 | load=123.18 |
| w1a | on1 | 6.331s | 4.49 | tok=774.05 | load=145.21 |
| w1a | off2 | 6.467s | 4.33 | tok=738.72 | load=157.91 |
| w1a | on2 | 6.214s | 4.45 | tok=777.48 | load=168.01 |
- 20:55Z BENCH1 verdict: emit+argmax group (#8/#9/#18) PROVISIONAL WINNER — ON beat OFF in 2/2 pairs on ALL metrics despite rising load (TTFT 6.8->6.33/6.47->6.21s; rps 4.27->4.49/4.33->4.45; tok 740->774/739->777, +4-5%). Dynflag toggle confirmed engaging. Cross-cell req-hash NOT valid for identity (off1!=off2 — closed-loop assignment nondeterministic); bench2 does B1 sequential identity check + #4 + #2 A/Bs (bench2.log, rows w1b). #3 NO-GO documented (async prep already shipped: PRE_PLAN_SPEC default-on; only remaining deferral = the V1 failure). #13 committed 1c4c25c3.
| w1b | ident_off | .299s | 0.88 | tok=173.81 | load=86.59 |
| w1b | ident_on | .36s | 0.88 | tok=172.80 | load=92.74 |
| w1b | identity | 1 | - | - | - |
| w1b | f4_off1 | 6.17s | 4.47 | tok=761.90 | load=117.08 |
| w1b | f4_on1 | 6.1s | 4.43 | tok=778.72 | load=132.10 |
| w1b | f4_off2 | 6.79s | 4.42 | tok=764.48 | load=147.97 |
| w1b | f4_on2 | 6.4s | 4.46 | tok=766.17 | load=157.67 |
| w1b | f2_off1 | 6.219s | 4.43 | tok=778.75 | load=156.33 |
| w1b | f2_on1 | 6.16s | 4.69 | tok=813.54 | load=168.07 |
| w1b | f2_off2 | 6.217s | 4.49 | tok=778.77 | load=154.44 |
| w1b | f2_on2 | 6.22s | 4.45 | tok=767.75 | load=140.76 |
- 21:06Z BENCH2 verdicts: IDENTITY_CHECK=1 (#18 byte-identical, cleared). #4 weak TTFT win (2/2 TTFT: -1%/-6%, rps/tok wash @load117-158) — quiet retest before ship. #2 inconclusive 1/2 (pair1 +5.9%rps/+4.5%tok, pair2 wash) — park unless quiet revives. CPUSET reboot launched (boot4, LAB_CPUSET=64-87,192-215; carries #6/#1/#13 code; node1 RAM recovered 94G); bench3 chains: cs_base x2 + SEQNUMS pair + COADMIT pair (BASE now includes w1a winners). Rows w3.
| w3 | cs_base1 | 6.234s | 4.24 | tok=727.83 | load=142.41 |
| w3 | cs_base2 | 7.821s | 3.89 | tok=674.78 | load=156.39 |
| w3 | seq_on1 | 3.98s | 6.47 | tok=1100.24 | load=123.31 |
| w3 | seq_off1 | .367s | 5.52 | tok=974.01 | load=110.28 |
| w3 | coad_on1 | 7.42s | 4.39 | tok=732.98 | load=104.54 |
| w3 | coad_off1 | 7.41s | 4.30 | tok=732.10 | load=99.62 |
| w3 | coad_on2 | 6.41s | 4.95 | tok=839.17 | load=141.60 |
- 21:43Z BENCH3 verdicts: #1 COADMIT wash at load 100-142 (7.42v7.41, park for quiet retest); #6 SEQNUMS unresolved (pair polluted: same-config cells 0.367s vs 7.41s = contention lottery); #14 CPUSET base wash vs unpinned. NOTE seq_off1 hit 367ms TTFT + 974 tok @load110 — DONE-bar TTFT proven possible under load. boot5 launched: MSTAR_DETOK_PROC=1 + CPUSET (#13 boot-pair); bench4 chains: dt_base x2 + s2t B32 n=128 guard + DONE-bar sniper (load<27, up to 2h). Rows w4.
| w4 | dt_base1 | .7s | 5.48 | tok=921.64 | load=98.20 |
| w4 | dt_base2 | 4.93s | 6.02 | tok=1046.47 | load=91.30 |
| w4 | dt_s2t_guard | .42s | 34.61 | tok=720.10 | load=83.34 |
- 22:04Z BENCH4 (detok boot): dt_base1 0.70s/5.48/922@98, dt_base2 4.93/6.02/1046@91 — both beat same-load prior-boot cells (7.41s/732@100): #13 POSITIVE-LEAN (lottery caveat). s2t guard PASS 0.42s/34.61/720@83 (no regression). DONE sniper hunting (load<27 gate; load 66 falling).
- 22:42Z Server died 22:24 (graceful SIGTERM ~30min post-boot, wrapper-trap pattern; NOT idle_guard — no KILL line, guard only WARNed 22:27). STRATEGY FLIP: no warm servers; watch_boot_run.sh launched (boots best stack DETOK+CPUSET only when load<27 falling x2 + GPUs free + RAM>=200G; runs up to 6 DONE cells; tears down after window; zero idle by construction; log fix20/window.log). bench4 chain killed.
| bisectcap | A_known | DIED | 22:04:52Z |
| bisectcap | B_merged | DIED | 22:16:28Z |
| bisectcache | E_owncache | READY | 22:34:34Z |
| bisectcache | F_sharedclean | SKIP_BUSY |
- 22:47Z ROOT CAUSE CONFIRMED: two-source inductor cache merge (fix20+godv9) = poisoned kernels = silent capture-phase native aborts (A/B/session1 deaths). E_owncache (per-worktree cache) READY in ~10min incl. 0.04s compile. FIX: per-worktree caches seeded by rsync from a same-code donor; NEVER merge caches across build dirs. Reaped 76860 orphaned tim shm segs. GPUs 6,7 taken by root sglang+Megatron 22:25 — session1b waits for next window (queues ready).
| bisect2 | G_noinfra | READY | 01:38:58Z |
| bisect2 | I_phases | READY | 02:05:59Z |
| bisect2 | J_mega | DIED | 02:11:16Z |
- 02:22Z BISECT2 VERDICT: G_noinfra READY, I_phases READY, J_mega DIED -> CULPRIT = MEGA_CACHE hook (kills capture even with NO artifact file — load path no-op — so bug is hook init at capture site in infra/boot-cache's cuda_graph_runner integration; serverlog fix20/bisect2_J_mega.serverlog). Ideas-all stack GPU-boots CLEAN. session1 relaunched with INFRA=phases-only (session1c).
- 02:40Z CORRECTION (megacache-fix agent): all tonight's 'capture deaths' were MY watchdogs killing healthy boots — capture on ideas-all full grids = 8-12min SILENT (J_mega log shows capture_done t=517/717s + ready + graceful shutdown; I/J serverlogs were the same boot; missing artifact = zero torch calls proven by CPU repro). Mega-cache EXONERATED, restored to session1. Cache-poisoning verdict also downgraded to UNPROVEN (A_known had a shorter 10-min window than E's 12.5). Boot7 stall guard relaxed 6->18min; ready windows already 35min. session1d watcher armed. LESSON: never watchdog a phase you haven't measured; BOOT_PHASES now gives real phase durations.
| b6 | A_i2t_B1 | ttft=?/? | itl=?/? | rps=? | tok=? | load=16.28 |
| b6 | A_i2t_B2 | ttft=?/? | itl=?/? | rps=? | tok=? | load=15.31 |
| b6 | A_i2t_B32 | ttft=?/? | itl=?/? | rps=? | tok=? | load=14.96 |
| b6 | A_s2t_B32 | ttft=?/? | itl=?/? | rps=? | tok=? | load=14.85 |
| b6 | A_coadeager_on | ttft=?/? | itl=?/? | rps=? | tok=? | load=14.58 |
| b6 | A_coadeager_off | ttft=?/? | itl=?/? | rps=? | tok=? | load=15.80 |
| b6 | A_encv2_on | ttft=?/? | itl=?/? | rps=? | tok=? | load=15.07 |
| b6 | A_encv2_off | ttft=?/? | itl=?/? | rps=? | tok=? | load=15.29 |
| b6 | A_wgd_on | ttft=?/? | itl=?/? | rps=? | tok=? | load=15.35 |
| b6 | A_wgd_off | ttft=?/? | itl=?/? | rps=? | tok=? | load=15.07 |
| b6 | A_slim_on | ttft=?/? | itl=?/? | rps=? | tok=? | load=22.93 |
| b6 | A_slim_off | ttft=?/? | itl=?/? | rps=? | tok=? | load=21.11 |
| b7 | B_i2t_B1 | ttft=0.093/0.137 | itl=0.004/0.005 | rps=1.13 | tok=207.67 | load=21.42 |
| b7 | B_i2t_B2 | ttft=0.104/0.134 | itl=0.005/0.007 | rps=1.96 | tok=339.73 | load=18.09 |
| b7 | B_i2t_B32 | ttft=0.164/2.382 | itl=0.015/0.041 | rps=8.99 | tok=1595.06 | load=19.86 |
| b7 | B_s2t_B32 | ttft=0.379/0.790 | itl=0.018/0.064 | rps=38.84 | tok=813.68 | load=18.66 |
| b7 | B_stack_on1 | ttft=?/? | itl=?/? | rps=? | tok=? | load=13.82 |
| b7 | B_stack_off1 | ttft=?/? | itl=?/? | rps=? | tok=? | load=14.21 |
| b7 | B_stack_on2 | ttft=?/? | itl=?/? | rps=? | tok=? | load=13.48 |
| b7 | B_stack_off2 | ttft=?/? | itl=?/? | rps=? | tok=? | load=13.63 |
| b7 | B_stack_B32_n96 | ttft=?/? | itl=?/? | rps=? | tok=? | load=13.88 |
| b6 | A_i2t_B1 | ttft=0.289/0.302 | itl=0.004/0.005 | rps=0.94 | tok=170.23 | load=32.77 |
| b6 | A_i2t_B2 | ttft=0.315/0.489 | itl=0.005/0.006 | rps=1.57 | tok=274.26 | load=32.85 |
| b6 | A_i2t_B32 | ttft=6.004/6.659 | itl=0.005/0.002 | rps=4.92 | tok=863.40 | load=65.79 |
| b6 | A_s2t_B32 | ttft=0.267/0.609 | itl=0.021/0.061 | rps=41.63 | tok=873.06 | load=49.83 |
| b6 | A_coadeager_on | ttft=5.739/5.841 | itl=0.004/0.006 | rps=4.73 | tok=824.27 | load=61.23 |
| b6 | A_coadeager_off | ttft=6.249/6.353 | itl=0.003/0.006 | rps=4.59 | tok=788.37 | load=74.80 |
| b6 | A_encv2_on | ttft=6.079/6.191 | itl=0.003/0.006 | rps=4.66 | tok=803.54 | load=118.12 |
| b6 | A_encv2_off | ttft=6.533/6.617 | itl=0.004/0.007 | rps=4.56 | tok=785.16 | load=103.58 |
| b6 | A_wgd_on | ttft=5.624/5.715 | itl=0.003/0.006 | rps=5.01 | tok=869.27 | load=87.79 |
| b6 | A_wgd_off | ttft=6.145/6.276 | itl=0.003/0.007 | rps=4.65 | tok=807.91 | load=86.63 |
| b6 | A_slim_on | ttft=5.924/6.040 | itl=0.004/0.006 | rps=4.75 | tok=804.60 | load=89.93 |
| b6 | A_slim_off | ttft=6.314/6.451 | itl=0.002/0.006 | rps=4.42 | tok=765.26 | load=87.81 |
| b7 | B_i2t_B1 | ttft=0.090/0.106 | itl=0.004/0.005 | rps=1.16 | tok=209.94 | load=21.82 |
| b7 | B_i2t_B2 | ttft=0.100/0.120 | itl=0.005/0.007 | rps=1.95 | tok=346.24 | load=20.58 |
| b7 | B_i2t_B32 | ttft=0.163/2.479 | itl=0.014/0.040 | rps=9.41 | tok=1654.37 | load=53.29 |
| b7 | B_s2t_B32 | ttft=0.301/0.695 | itl=0.021/0.055 | rps=39.70 | tok=831.47 | load=38.34 |
| b7 | B_stack_on1 | ttft=1.403/2.654 | itl=0.012/0.047 | rps=8.01 | tok=1404.55 | load=42.50 |
| b7 | B_stack_off1 | ttft=0.996/2.486 | itl=0.013/0.042 | rps=8.04 | tok=1401.36 | load=41.33 |
| b7 | B_stack_on2 | ttft=1.101/2.768 | itl=0.013/0.037 | rps=7.82 | tok=1357.93 | load=29.38 |
| b7 | B_stack_off2 | ttft=1.131/2.630 | itl=0.014/0.043 | rps=7.67 | tok=1328.90 | load=37.49 |
| b7 | B_stack_B32_n96 | ttft=0.195/2.417 | itl=0.016/0.041 | rps=8.53 | tok=1511.46 | load=48.88 |
