"""
One command to reproduce every result and figure.

    python run_all.py            fetch anything not cached, then run all stages
    python run_all.py --offline  use data/ and results/mp_ml_dataset.csv only
                                 (no network; skips the Phase-1 ML refit)
    python run_all.py --skip-ml  keep existing Phase-1 ML results (~1.5 h to refit)

Seeds are fixed in every script (SEED = 42). MP_API_KEY is read from the
environment, never from code.
"""

import argparse
import os
import subprocess
import sys
import time

STEPS = [
    ("MP data", ["adib_data.py"], "net"),
    ("literature table", ["literature_benchmark.py"], None),
    ("Phase-1 ML benchmark", ["substrate_screening.py", "--key", "{KEY}",
                              "--out", "results/model_comparison.csv",
                              "--audit-out", "results/leakage_audit.csv",
                              "--oof-out", "results/oof_predictions.csv",
                              "--data-out", "results/mp_ml_dataset.csv"], "ml"),
    ("screening", ["adib_screening.py"], None),
    ("validation", ["adib_validate.py"], None),
    ("shortlist", ["adib_shortlist.py"], None),
    ("ML applicability domain", ["adib_ml_domain.py"], None),
    ("figures", ["make_figures.py"], None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--skip-ml", action="store_true")
    args = ap.parse_args()
    key = os.environ.get("MP_API_KEY", "")
    for name, cmd, kind in STEPS:
        if kind == "net" and args.offline:
            print(f"-- {name}: skipped (offline)")
            continue
        if kind == "ml" and (args.offline or args.skip_ml):
            print(f"-- {name}: skipped (using existing results/)")
            continue
        if "{KEY}" in cmd and not key:
            sys.exit("MP_API_KEY not set; use --offline or --skip-ml.")
        cmd = [sys.executable, "-W", "ignore"] + [c.replace("{KEY}", key) for c in cmd]
        t = time.time()
        print(f"== {name}", flush=True)
        if subprocess.run(cmd).returncode:
            sys.exit(f"step '{name}' failed")
        print(f"-- {name}: {time.time() - t:.0f}s\n", flush=True)


if __name__ == "__main__":
    main()
