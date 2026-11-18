"""
One-time conversion of the Kaggle M5 files into results/m5_foods_daily.npz.

    python -m experiments.prepare_m5 --sales sales_train_evaluation.csv --calendar calendar.csv

Then run any experiment with real demand by setting SCTWIN_M5:
    SCTWIN_M5=results/m5_foods_daily.npz python -m experiments.run_baseline --reps 200
"""
import argparse, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from twin import m5


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sales", required=True)
    ap.add_argument("--calendar", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "m5_foods_daily.npz"))
    a = ap.parse_args()
    path = m5.prepare(a.sales, a.calendar, a.out)
    m5.enable(path)
    z = np.load(path)
    print(f"saved {path}: stores {list(z['stores'])}, {z['sales'].shape[1]} days")
    for ours, real in m5.STORE_MAP.items():
        s = z["sales"][list(z["stores"]).index(real)]
        print(f"  {ours} <- {real}: mean {s.mean():.0f} units/day, CV {s.std()/s.mean():.2f}, zero days {(s == 0).sum()}")
    print(f"{m5.n_windows()} one-year windows available")
