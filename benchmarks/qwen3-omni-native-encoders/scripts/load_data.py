"""Shared data loader for verification scripts.

Reads raw_<path>.json from the benchmark directory (sibling of scripts/).
All verification scripts import this instead of calling git show.
"""
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..")

def load_raw(path_name):
    """Load raw benchmark JSON for a given path (e.g. 'audio_to_text')."""
    fpath = os.path.join(DATA_DIR, f"raw_{path_name}.json")
    with open(fpath) as f:
        return json.load(f)
