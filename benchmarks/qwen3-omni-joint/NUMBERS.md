# NUMBERS.md -- headline numbers (auto-generated from clean sweep data 2026-06-29)

Single source of truth. M*-new data: integration-mnew build f58a805, flags MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_CHUNKED_PREFILL=1.
M*-new-combined: same + MSTAR_VISION_GRAPH_ALIGN=1 MSTAR_BATCH_VISION_PREFILL=1 (commit e943d72).
M*-old data: upstream main ae7d173, no optimization flags.
Ratios: combined/vLLM and combined/old; for lower-is-better metrics the ratio is inverted so **>1.00x always means M*-new is faster**.

## S2T -- audio_to_text

throughput unit = **tok/s**; req/s / TTFT(text) p50 / ITL(text) mean

| B | metric | M*-combined | M*-new | M*-old | vLLM | comb/new | comb/vLLM | comb/old |
|---|---|---|---|---|---|---|---|---|
| 1 | throughput | 71.6531 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 1 | req_per_s | 5.0602 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 1 | TTFT_text_p50 | 0.1006 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 1 | ITL_text_mean | 0.0070 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | throughput | 97.2316 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | req_per_s | 6.9253 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | TTFT_text_p50 | 0.1434 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | ITL_text_mean | 0.0109 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | throughput | 126.1287 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | req_per_s | 8.9200 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | TTFT_text_p50 | 0.1855 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | ITL_text_mean | 0.0186 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | throughput | 176.6716 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | req_per_s | 12.1633 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | TTFT_text_p50 | 0.2480 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | ITL_text_mean | 0.0296 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | throughput | 255.5254 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | req_per_s | 17.9002 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | TTFT_text_p50 | 0.2883 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | ITL_text_mean | 0.0430 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | throughput | 337.0164 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | req_per_s | 23.5316 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | TTFT_text_p50 | 0.4202 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | ITL_text_mean | 0.0647 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |

## I2T -- image_to_text

throughput unit = **tok/s**; req/s / TTFT(text) p50 / ITL(text) mean

| B | metric | M*-combined | M*-new | M*-old | vLLM | comb/new | comb/vLLM | comb/old |
|---|---|---|---|---|---|---|---|---|
| 1 | throughput | 120.0733 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 1 | req_per_s | 0.6934 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 1 | TTFT_text_p50 | 0.2825 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 1 | ITL_text_mean | 0.0068 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | throughput | 186.1150 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | req_per_s | 1.0952 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | TTFT_text_p50 | 0.3246 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | ITL_text_mean | 0.0088 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | throughput | 292.1021 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | req_per_s | 1.7272 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | TTFT_text_p50 | 0.3487 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | ITL_text_mean | 0.0113 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | throughput | 423.7843 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | req_per_s | 2.4376 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | TTFT_text_p50 | 0.4171 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | ITL_text_mean | 0.0157 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | throughput | 571.6056 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | req_per_s | 3.2994 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | TTFT_text_p50 | 0.4999 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | ITL_text_mean | 0.0237 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | throughput | 740.7304 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | req_per_s | 4.2092 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | TTFT_text_p50 | 0.5846 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | ITL_text_mean | 0.0377 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |

## I2S -- image_to_speech

throughput unit = **audio sec/s**; TTFT(audio) p50 / ITL(audio) mean

| B | metric | M*-combined | M*-new | M*-old | vLLM | comb/new | comb/vLLM | comb/old |
|---|---|---|---|---|---|---|---|---|
| 1 | req_per_s | 0.2767 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 1 | TTFT_audio_p50 | 0.4012 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 1 | ITL_audio_mean | 0.0914 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | req_per_s | 0.4577 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | TTFT_audio_p50 | 0.4710 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | ITL_audio_mean | 0.1136 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | req_per_s | 0.7348 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | TTFT_audio_p50 | 0.5494 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | ITL_audio_mean | 0.1364 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | req_per_s | 1.1465 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | TTFT_audio_p50 | 0.6560 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | ITL_audio_mean | 0.1624 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | req_per_s | 1.7234 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | TTFT_audio_p50 | 0.7974 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | ITL_audio_mean | 0.2168 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | req_per_s | 2.2184 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | TTFT_audio_p50 | 1.0741 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | ITL_audio_mean | 0.3416 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |

## S2S -- audio_to_speech

throughput unit = **audio sec/s**; TTFT(audio) p50 / ITL(audio) mean

| B | metric | M*-combined | M*-new | M*-old | vLLM | comb/new | comb/vLLM | comb/old |
|---|---|---|---|---|---|---|---|---|
| 1 | req_per_s | 2.7260 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 1 | TTFT_audio_p50 | 0.2244 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 1 | ITL_audio_mean | 0.0668 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | req_per_s | 4.2849 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | TTFT_audio_p50 | 0.3023 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 2 | ITL_audio_mean | 0.0835 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | req_per_s | 5.9148 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | TTFT_audio_p50 | 0.4251 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 4 | ITL_audio_mean | 0.1022 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | req_per_s | 8.8005 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | TTFT_audio_p50 | 0.6080 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 8 | ITL_audio_mean | 0.1238 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | req_per_s | 12.7308 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | TTFT_audio_p50 | 0.8322 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 16 | ITL_audio_mean | 0.1528 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | req_per_s | 18.4827 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | TTFT_audio_p50 | 1.3001 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
| 32 | ITL_audio_mean | 0.2183 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a |
