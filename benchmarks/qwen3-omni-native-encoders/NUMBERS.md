# NUMBERS.md -- headline numbers (auto-generated from clean sweep data 2026-06-29)

Single source of truth. M*-new data: integration-mnew build f58a805, flags MSTAR_GPU_MEL=1 MSTAR_GPU_IMAGE_PREPROCESS=1 MSTAR_CHUNKED_PREFILL=1.
M*-old data: upstream main ae7d173, no optimization flags.
Ratios: new/vLLM and new/old; for lower-is-better metrics the ratio is inverted so **>1.00x always means M*-new is faster**.

## S2T -- audio_to_text

throughput unit = **tok/s**; req/s / TTFT(text) p50 / ITL(text) mean

| B | metric | M*-new | M*-old | vLLM | new/vLLM | new/old |
|---|---|---|---|---|---|---|
| 1 | throughput | 69.7439 | 27.6892 | 56.9672 | 1.22x | 2.52x |
| 1 | req_per_s | 4.9324 | 1.9122 | 2.3008 | 2.14x | 2.58x |
| 1 | TTFT_text_p50 | 0.1016 | 0.3910 | 0.1400 | 1.38x | 3.85x |
| 1 | ITL_text_mean | 0.0072 | 0.0074 | 0.0120 | 1.67x | 1.04x |
| 2 | throughput | 96.0318 | 42.1342 | 80.0659 | 1.20x | 2.28x |
| 2 | req_per_s | 6.8399 | 2.9098 | 3.2337 | 2.12x | 2.35x |
| 2 | TTFT_text_p50 | 0.1439 | 0.7245 | 0.1320 | 0.92x | 5.03x |
| 2 | ITL_text_mean | 0.0112 | 0.0007 | 0.0200 | 1.78x | 0.06x |
| 4 | throughput | 113.0369 | 39.2264 | 105.4143 | 1.07x | 2.88x |
| 4 | req_per_s | 7.9491 | 2.7090 | 4.2574 | 1.87x | 2.93x |
| 4 | TTFT_text_p50 | 0.1946 | 1.2417 | 0.1500 | 0.77x | 6.38x |
| 4 | ITL_text_mean | 0.0216 | 0.0030 | 0.0320 | 1.48x | 0.14x |
| 8 | throughput | 178.2386 | 36.5444 | 132.6499 | 1.34x | 4.88x |
| 8 | req_per_s | 11.9925 | 2.5225 | 5.5041 | 2.18x | 4.75x |
| 8 | TTFT_text_p50 | 0.2437 | 2.8495 | 0.2300 | 0.94x | 11.69x |
| 8 | ITL_text_mean | 0.0291 | 0.0005 | 0.0500 | 1.72x | 0.02x |
| 16 | throughput | 227.7122 | 35.1537 | 189.2391 | 1.20x | 6.48x |
| 16 | req_per_s | 14.3610 | 2.4497 | 7.9554 | 1.81x | 5.86x |
| 16 | TTFT_text_p50 | 0.3211 | 6.2252 | 0.2360 | 0.73x | 19.39x |
| 16 | ITL_text_mean | 0.0434 | 0.0002 | 0.0670 | 1.54x | 0.00x |
| 32 | throughput | 325.8489 | 35.2157 | 305.0155 | 1.07x | 9.25x |
| 32 | req_per_s | 22.7568 | 2.4477 | 12.6677 | 1.80x | 9.30x |
| 32 | TTFT_text_p50 | 0.4556 | 12.9031 | 0.2830 | 0.62x | 28.32x |
| 32 | ITL_text_mean | 0.0686 | 0.0091 | 0.0770 | 1.12x | 0.13x |

## I2T -- image_to_text

throughput unit = **tok/s**; req/s / TTFT(text) p50 / ITL(text) mean

| B | metric | M*-new | M*-old | vLLM | new/vLLM | new/old |
|---|---|---|---|---|---|---|
| 1 | throughput | 115.7432 | 109.5898 | 77.2192 | 1.50x | 1.06x |
| 1 | req_per_s | 0.6926 | 0.6349 | 0.3727 | 1.86x | 1.09x |
| 1 | TTFT_text_p50 | 0.2964 | 0.4010 | 0.1510 | 0.51x | 1.35x |
| 1 | ITL_text_mean | 0.0069 | 0.0070 | 0.0120 | 1.74x | 1.01x |
| 2 | throughput | 181.3224 | 167.3596 | 110.1704 | 1.65x | 1.08x |
| 2 | req_per_s | 1.0852 | 0.9690 | 0.5301 | 2.05x | 1.12x |
| 2 | TTFT_text_p50 | 0.3297 | 0.4480 | 0.1350 | 0.41x | 1.36x |
| 2 | ITL_text_mean | 0.0089 | 0.0092 | 0.0170 | 1.90x | 1.03x |
| 4 | throughput | 276.8928 | 253.1901 | 149.9655 | 1.85x | 1.09x |
| 4 | req_per_s | 1.6448 | 1.4765 | 0.7122 | 2.31x | 1.11x |
| 4 | TTFT_text_p50 | 0.3624 | 0.5197 | 0.1510 | 0.42x | 1.43x |
| 4 | ITL_text_mean | 0.0119 | 0.0123 | 0.0260 | 2.18x | 1.04x |
| 8 | throughput | 419.5322 | 348.5283 | 211.9746 | 1.98x | 1.20x |
| 8 | req_per_s | 2.3291 | 2.0361 | 1.0080 | 2.31x | 1.14x |
| 8 | TTFT_text_p50 | 0.3823 | 0.5401 | 0.1870 | 0.49x | 1.41x |
| 8 | ITL_text_mean | 0.0161 | 0.0185 | 0.0360 | 2.24x | 1.15x |
| 16 | throughput | 572.4369 | 466.4692 | 320.6194 | 1.79x | 1.23x |
| 16 | req_per_s | 3.2334 | 2.7368 | 1.5154 | 2.13x | 1.18x |
| 16 | TTFT_text_p50 | 0.4778 | 0.6884 | 0.1910 | 0.40x | 1.44x |
| 16 | ITL_text_mean | 0.0235 | 0.0276 | 0.0470 | 2.00x | 1.18x |
| 32 | throughput | 704.8083 | 565.9690 | 531.0198 | 1.33x | 1.25x |
| 32 | req_per_s | 4.0330 | 3.2168 | 2.5379 | 1.59x | 1.25x |
| 32 | TTFT_text_p50 | 0.6666 | 1.0291 | 0.2050 | 0.31x | 1.54x |
| 32 | ITL_text_mean | 0.0381 | 0.0453 | 0.0570 | 1.50x | 1.19x |

## I2S -- image_to_speech

throughput unit = **audio sec/s**; RTF p50 / TTFT(audio) p50 / ITL(audio) mean

| B | metric | M*-new | M*-old | vLLM | new/vLLM | new/old |
|---|---|---|---|---|---|---|
| 1 | throughput | 11.5266 | 11.0862 | 6.3860 | 1.80x | 1.04x |
| 1 | RTF_p50 | 0.0859 | 0.0895 | 0.1568 | 1.83x | 1.04x |
| 1 | TTFT_audio_p50 | 0.4100 | 0.6255 | 0.5580 | 1.36x | 1.53x |
| 1 | ITL_audio_mean | 0.0920 | 0.1547 | 0.2970 | 3.23x | 1.68x |
| 2 | throughput | 18.1547 | 17.6750 | 9.9058 | 1.83x | 1.03x |
| 2 | RTF_p50 | 0.1091 | 0.1120 | 0.2001 | 1.84x | 1.03x |
| 2 | TTFT_audio_p50 | 0.4652 | 0.7701 | 0.8510 | 1.83x | 1.66x |
| 2 | ITL_audio_mean | 0.1164 | 0.1918 | 0.3760 | 3.23x | 1.65x |
| 4 | throughput | 30.2713 | 29.3348 | 15.8497 | 1.91x | 1.03x |
| 4 | RTF_p50 | 0.1296 | 0.1363 | 0.2497 | 1.93x | 1.05x |
| 4 | TTFT_audio_p50 | 0.5569 | 0.9008 | 1.0520 | 1.89x | 1.62x |
| 4 | ITL_audio_mean | 0.1376 | 0.2309 | 0.4640 | 3.37x | 1.68x |
| 8 | throughput | 51.0081 | 51.7271 | 23.7847 | 2.14x | 0.99x |
| 8 | RTF_p50 | 0.1522 | 0.1528 | 0.3269 | 2.15x | 1.00x |
| 8 | TTFT_audio_p50 | 0.6078 | 0.9272 | 1.3200 | 2.17x | 1.53x |
| 8 | ITL_audio_mean | 0.1625 | 0.2643 | 0.6130 | 3.77x | 1.63x |
| 16 | throughput | 73.8330 | 73.8076 | 34.8718 | 2.12x | 1.00x |
| 16 | RTF_p50 | 0.2096 | 0.2111 | 0.4517 | 2.15x | 1.01x |
| 16 | TTFT_audio_p50 | 0.8034 | 1.2577 | 1.7680 | 2.20x | 1.57x |
| 16 | ITL_audio_mean | 0.2206 | 0.3575 | 0.8390 | 3.80x | 1.62x |
| 32 | throughput | 91.4911 | 92.8256 | 47.8527 | 1.91x | 0.99x |
| 32 | RTF_p50 | 0.3401 | 0.3370 | 0.6550 | 1.93x | 0.99x |
| 32 | TTFT_audio_p50 | 1.0727 | 1.8056 | 2.4940 | 2.32x | 1.68x |
| 32 | ITL_audio_mean | 0.3629 | 0.5724 | 1.2270 | 3.38x | 1.58x |

## S2S -- audio_to_speech

throughput unit = **audio sec/s**; RTF p50 / TTFT(audio) p50 / ITL(audio) mean

| B | metric | M*-new | M*-old | vLLM | new/vLLM | new/old |
|---|---|---|---|---|---|---|
| 1 | throughput | 8.1267 | 4.2319 | 5.5858 | 1.45x | 1.92x |
| 1 | RTF_p50 | 0.1261 | 0.2288 | 0.1886 | 1.50x | 1.81x |
| 1 | TTFT_audio_p50 | 0.2264 | 0.5988 | 0.5340 | 2.36x | 2.64x |
| 1 | ITL_audio_mean | 0.0640 | 0.0851 | 0.2390 | 3.73x | 1.33x |
| 2 | throughput | 12.9671 | 7.3188 | 8.8127 | 1.47x | 1.77x |
| 2 | RTF_p50 | 0.1540 | 0.2964 | 0.2431 | 1.58x | 1.93x |
| 2 | TTFT_audio_p50 | 0.2838 | 0.8374 | 0.7100 | 2.50x | 2.95x |
| 2 | ITL_audio_mean | 0.0810 | 0.0382 | 0.2730 | 3.37x | 0.47x |
| 4 | throughput | 18.2919 | 7.1647 | 13.7993 | 1.33x | 2.55x |
| 4 | RTF_p50 | 0.2209 | 0.5334 | 0.3055 | 1.38x | 2.41x |
| 4 | TTFT_audio_p50 | 0.4252 | 1.2781 | 0.9110 | 2.14x | 3.01x |
| 4 | ITL_audio_mean | 0.1040 | 0.1826 | 0.3430 | 3.30x | 1.76x |
| 8 | throughput | 29.0064 | 7.5456 | 19.5556 | 1.48x | 3.84x |
| 8 | RTF_p50 | 0.2813 | 1.0780 | 0.3948 | 1.40x | 3.83x |
| 8 | TTFT_audio_p50 | 0.5627 | 2.9416 | 1.2470 | 2.22x | 5.23x |
| 8 | ITL_audio_mean | 0.1254 | 0.0865 | 0.4340 | 3.46x | 0.69x |
| 16 | throughput | 38.9073 | 7.3459 | 24.1285 | 1.61x | 5.30x |
| 16 | RTF_p50 | 0.3993 | 2.0912 | 0.5336 | 1.34x | 5.24x |
| 16 | TTFT_audio_p50 | 0.8231 | 6.2094 | 1.6740 | 2.03x | 7.54x |
| 16 | ITL_audio_mean | 0.1923 | 0.1428 | 0.5560 | 2.89x | 0.74x |
| 32 | throughput | 53.3701 | 8.4385 | 33.5188 | 1.59x | 6.32x |
| 32 | RTF_p50 | 0.6029 | 4.2972 | 0.7783 | 1.29x | 7.13x |
| 32 | TTFT_audio_p50 | 1.3353 | 12.8144 | 2.3160 | 1.73x | 9.60x |
| 32 | ITL_audio_mean | 0.2243 | 0.1454 | 0.8520 | 3.80x | 0.65x |

