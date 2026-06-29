#!/usr/bin/env python3
"""Compare audio outputs between two sweep runs (binary + duration + waveform similarity).

Usage:
  python compare_audio.py \
      --reference /home/tim/tmp/mstar_old_s2s_audio_reference \
      --candidate /home/tim/tmp/sweep_mnew_s2s_rerun/s2s \
      --output comparison_report.json

For each batch size, compares req_N.wav files:
  - Binary identical? (exact byte match)
  - Duration match? (same number of samples)
  - Waveform similarity (cosine similarity of float32 samples)
  - Peak amplitude difference

Prints a summary table and writes detailed JSON.
"""
import argparse, json, os, struct, wave, math


def read_wav_samples(path):
    """Read a wav file and return (sample_rate, samples_as_floats)."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nf = w.getnframes()
        nc = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(nf)

    if sw == 2:
        fmt = f"<{nf * nc}h"
        ints = struct.unpack(fmt, raw)
        samples = [x / 32768.0 for x in ints]
    elif sw == 4:
        fmt = f"<{nf * nc}i"
        ints = struct.unpack(fmt, raw)
        samples = [x / 2147483648.0 for x in ints]
    else:
        samples = list(raw)

    return sr, samples


def cosine_similarity(a, b):
    if len(a) != len(b) or len(a) == 0:
        return None
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-12 or nb < 1e-12:
        return None
    return dot / (na * nb)


def compare_one(ref_path, cand_path):
    result = {"ref": ref_path, "cand": cand_path}

    with open(ref_path, "rb") as f:
        ref_bytes = f.read()
    with open(cand_path, "rb") as f:
        cand_bytes = f.read()

    result["binary_identical"] = ref_bytes == cand_bytes
    result["ref_size"] = len(ref_bytes)
    result["cand_size"] = len(cand_bytes)

    try:
        sr_r, samp_r = read_wav_samples(ref_path)
        sr_c, samp_c = read_wav_samples(cand_path)
        result["ref_sr"] = sr_r
        result["cand_sr"] = sr_c
        result["ref_samples"] = len(samp_r)
        result["cand_samples"] = len(samp_c)
        result["duration_match"] = len(samp_r) == len(samp_c)

        min_len = min(len(samp_r), len(samp_c))
        if min_len > 0:
            result["cosine_sim"] = cosine_similarity(samp_r[:min_len], samp_c[:min_len])
            diffs = [abs(a - b) for a, b in zip(samp_r[:min_len], samp_c[:min_len])]
            result["max_abs_diff"] = max(diffs)
            result["mean_abs_diff"] = sum(diffs) / len(diffs)
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True, help="Reference dir (M*-old audio)")
    p.add_argument("--candidate", required=True, help="Candidate dir (M*-new audio)")
    p.add_argument("--output", default="comparison_report.json")
    p.add_argument("--batches", default="1,2,4,8,16,32")
    args = p.parse_args()

    batches = [int(x) for x in args.batches.split(",")]
    report = {"reference": args.reference, "candidate": args.candidate, "batches": {}}

    print(f"{'B':>3} | {'files':>5} | {'identical':>9} | {'dur_match':>9} | "
          f"{'cos_sim_min':>11} | {'cos_sim_mean':>12} | {'max_diff':>8}")
    print("-" * 80)

    for b in batches:
        ref_dir = os.path.join(args.reference, f"B{b}")
        cand_dir = os.path.join(args.candidate, f"B{b}")

        if not os.path.isdir(ref_dir):
            print(f"{b:>3} | SKIP (no ref dir)")
            continue
        if not os.path.isdir(cand_dir):
            print(f"{b:>3} | SKIP (no cand dir)")
            continue

        ref_wavs = sorted([f for f in os.listdir(ref_dir) if f.endswith(".wav")])
        cand_wavs = sorted([f for f in os.listdir(cand_dir) if f.endswith(".wav")])
        common = sorted(set(ref_wavs) & set(cand_wavs))

        comparisons = []
        identical = 0
        dur_match = 0
        cos_sims = []
        max_diffs = []

        for fname in common:
            c = compare_one(os.path.join(ref_dir, fname), os.path.join(cand_dir, fname))
            comparisons.append(c)
            if c.get("binary_identical"):
                identical += 1
            if c.get("duration_match"):
                dur_match += 1
            if c.get("cosine_sim") is not None:
                cos_sims.append(c["cosine_sim"])
            if c.get("max_abs_diff") is not None:
                max_diffs.append(c["max_abs_diff"])

        report["batches"][f"B{b}"] = {
            "n_compared": len(common),
            "n_identical": identical,
            "n_duration_match": dur_match,
            "cos_sim_min": min(cos_sims) if cos_sims else None,
            "cos_sim_mean": sum(cos_sims) / len(cos_sims) if cos_sims else None,
            "max_abs_diff": max(max_diffs) if max_diffs else None,
        }

        cs_min = f"{min(cos_sims):.6f}" if cos_sims else "—"
        cs_mean = f"{sum(cos_sims)/len(cos_sims):.6f}" if cos_sims else "—"
        md = f"{max(max_diffs):.6f}" if max_diffs else "—"
        print(f"{b:>3} | {len(common):>5} | {identical:>9} | {dur_match:>9} | "
              f"{cs_min:>11} | {cs_mean:>12} | {md:>8}")

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report: {args.output}")


if __name__ == "__main__":
    main()
