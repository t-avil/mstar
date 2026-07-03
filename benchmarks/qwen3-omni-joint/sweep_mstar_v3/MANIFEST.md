# sweep_mstar_v3 — final-stack preview cells (2026-07-03, GPU pair 0,1)
CAVEAT: pair 0,1 is cross-NUMA under the harness's node-1 binding —
absolutes read ~10% below the canonical pair 6,7. Canonical sweep pending
(claim watcher armed). Every source run sentinel-validated + co-location
audited. Flags: final stack (see BEATING_NEW_VLLM.md §2 + MSTAR_CONDUCTOR_POLL).
Provenance: i2t B1/B8, s2t B8/B32, s2s B8, i2s B8 <- qb_finalsweep;
i2t B2/B4/B16/B32 <- qb_midbatch2 (sentinels 6.303/6.559).
