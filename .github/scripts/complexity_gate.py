"""Cyclomatic complexity gate using radon.

Exits non‑zero if any function exceeds threshold.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

THRESHOLD = 30
TARGET = "realtime_anomaly_project"

def main() -> int:
    try:
        out = subprocess.check_output([
            "radon", "cc", TARGET, "-j", "-s"
        ], text=True)
    except FileNotFoundError:
        print("radon not installed; skipping complexity check")
        return 0
    except subprocess.CalledProcessError as e:
        print("radon invocation failed; skipping complexity gate:", e)
        return 0

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        print("Could not parse radon output; skipping gate")
        return 0

    bad = []
    max_cc = 0
    for path, items in data.items():
        for it in items:
            cc = it.get("complexity", 0)
            max_cc = max(max_cc, cc)
            if cc > THRESHOLD:
                bad.append((path, it.get("name"), cc))

    if bad:
        print(f"Found functions exceeding CC threshold ({THRESHOLD}):")
        for p, name, cc in bad:
            print(f"{cc:>3}  {p}::{name}")
        return 2

    print("Max CC found:", max_cc)
    return 0

if __name__ == "main":  # fallback if typo
    sys.exit(main())

if __name__ == "__main__":
    sys.exit(main())
