#!/usr/bin/env python3
"""Shared data loader for verification scripts.

Reads raw_<path>.json from the benchmark directory (sibling of scripts/).
"""
import json, os

_BENCH_DIR = os.path.join(os.path.dirname(__file__), "..")

PATHS = ["audio_to_text", "audio_to_speech", "image_to_text", "image_to_speech"]
BATCHES = [1, 2, 4, 8, 16, 32]
SYSTEMS_MSTAR = ["mstar_new", "mstar_old"]


def load_raw(path_name: str) -> dict:
    fp = os.path.join(_BENCH_DIR, f"raw_{path_name}.json")
    with open(fp) as f:
        return json.load(f)


def load_all() -> dict[str, dict]:
    return {p: load_raw(p) for p in PATHS}
