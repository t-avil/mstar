---
name: mstar-lab-wrapper-reaping
description: lab_server.sh wrapper dies when the agent shell tree is reaped between turns — setsid it
metadata: 
  node_type: memory
  type: project
  originSessionId: c8db2eed-2663-4432-80ce-915cc10be606
---

lab_server.sh boots the M* server via `setsid ... &` but the WRAPPER script
itself keeps running (readiness loop, warm cell) inside the agent's shell
process tree, holding a `trap cleanup INT TERM`. When the harness reaps that
shell tree between agent turns, the wrapper gets SIGTERM, the trap fires, and it
kills the healthy server's process group. Signature in server.log: a graceful
uvicorn shutdown ("INFO: Shutting down / Application shutdown complete /
Finished server process") with NO fault/OOM/traceback, followed later by
`emit_sidecar: worker ... is gone; exiting`. The server was fine — an external
TERM to the wrapper killed it.

Confirmed the mechanism behind multiple "mystery" warm-server deaths (2026-07-03
and 2026-07-04 night runs): a server reached WARM+READY, sat idle, then died at
a turn boundary with that exact graceful-shutdown signature.

**Why:** the wrapper's own lifetime is coupled to the agent shell tree, so a
turn-boundary reap propagates TERM into the trap and reaps the server too.

**How to apply:** when launching lab_server.sh (or any wrapper that traps
TERM to reap a long-lived child) from an agent shell, detach the WRAPPER from
the agent tree — e.g. `setsid lab_server.sh ...` / a fresh session — so a
turn-boundary reap can't deliver TERM to it. Also: after a boot hits
WARM+READY, fire the lab_ab cells immediately in the same active flow; don't
end the turn on a passive readiness watcher, or the warm server sits idle and
is exposed to exactly this reap. See [[mstar-bench-env-quirks]].
