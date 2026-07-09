
## Flag decomposition on the sidecar stack (labs, 3 rounds each) — the stack re-baselined
pair_off (cache+checkstop both OFF vs ON, lab1/6,7): 0.941/1.025/0.993 =
geomean -1.4% -> the pair retains ~0-3%, NOT the historical +7-10%; that
claim is RETRACTED for the current stack (pre-sidecar reality vs window
inflation indistinguishable retroactively). Flags stay ON (harmless, small
positive direction). route_decomp (FAST_ROUTE OFF vs ON, lab2/2,3):
0.958/0.864/1.001 = geomean -6.1% -> FAST_ROUTE STILL CONVERTS (+~6.5%)
post-sidecar — its work (route memoization) lives on the worker main thread,
which the sidecar did not move. Cross-pair confirmation swaps queued
(route on lab1, pair on lab2). Lesson: every landed flag's win must be
re-decomposed after a structural change (sidecar); "the stack" is not a sum
of its historical deltas.
