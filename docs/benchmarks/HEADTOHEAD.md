# HEADTOHEAD.md — live same-window A/B vs vLLM 0.22 (supplementary)

2026-07-02, interleaved (both servers preloaded, per-cell back-to-back);
vLLM on GPUs 4,5 / M*-v2 (encoff config) on 6,7. n=20/80/192 for B1/8/32.
Note: live vLLM measures ABOVE its committed baselines on B1/B8 cells
(committed run had contention); B32 matches committed. Output-length skew:
vLLM generates ~20-25% more tokens per request on identical inputs.

- i2t B=1  vllm=0.886 req/s (190.6 tok/s)  v2=0.779 req/s (134.4 tok/s)  v2/vllm=0.879x
- i2t B=8  vllm=3.401 req/s (728.8 tok/s)  v2=3.283 req/s (589.2 tok/s)  v2/vllm=0.965x
- i2t B=32  vllm=8.326 req/s (1744.9 tok/s)  v2=6.101 req/s (1074.3 tok/s)  v2/vllm=0.733x
- s2t B=1  vllm=4.312 req/s (139.5 tok/s)  v2=3.853 req/s (77.5 tok/s)  v2/vllm=0.894x
- s2t B=8  vllm=15.130 req/s (366.0 tok/s)  v2=15.854 req/s (334.7 tok/s)  v2/vllm=1.048x
- s2t B=32  vllm=30.832 req/s (728.6 tok/s)  v2=31.827 req/s (671.4 tok/s)  v2/vllm=1.032x
