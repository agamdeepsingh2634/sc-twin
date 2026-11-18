"""
Transfer test: does a policy optimised on SYNTHETIC demand still work on REAL
M5 demand? Evaluates both knee policies (synthetic-optimised and M5-optimised)
on the same 210 held-out M5 runs.  Output: results/m5/transfer_test.csv
"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SCTWIN_M5", os.path.join(ROOT, "results", "m5_foods_daily.npz"))
from twin.config import Policy
from twin.optimize import make_cases, evaluate_policy
from experiments.analyze_optimization import test_front, knee


def knee_from(folder, name):
    k = knee(test_front(pd.read_csv(os.path.join(folder, "opt_pareto_test.csv"))))
    return Policy(name, store=(k.store_s, k.store_S), dc=(k.dc_s, k.dc_S),
                  fg=(k.fg_s, k.fg_S), rm=(k.rm_s, k.rm_S))


if __name__ == "__main__":
    cases = make_cases(30, base_seed=900_000)
    rows = []
    for folder, name in [(os.path.join(ROOT, "results"), "Knee optimised on synthetic"),
                         (os.path.join(ROOT, "results", "m5"), "Knee optimised on real M5")]:
        m = evaluate_policy(knee_from(folder, name), cases)
        rows.append(dict(policy=name, total_L=m["cost_total"] / 1e5, op_L=m["cost_operating"] / 1e5,
                         lost_pct=100 * (1 - m["fill_rate"]), recovery=m["recovery_time"]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "m5", "transfer_test.csv"), index=False)
    print(df.round(2).to_string())
