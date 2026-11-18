"""Compare the optimised Pareto frontier with the reference strategies
(out-of-sample test results) -> results/opt_key_findings.csv"""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from twin.paths import results_dir
RES = results_dir(ROOT)


def test_front(p):
    """Keep only solutions that are still non-dominated on TEST data."""
    f = p[["test_cost_L", "test_lost_pct"]].values
    keep = [not np.any(np.all(f <= f[i], 1) & np.any(f < f[i], 1)) for i in range(len(f))]
    return p[keep].sort_values("test_cost_L").reset_index(drop=True)


def knee(p):
    f = p[["test_cost_L", "test_lost_pct"]].values
    n = (f - f.min(0)) / (f.max(0) - f.min(0))
    a, b = n[0], n[-1]
    ab = b - a
    d = np.abs(ab[0] * (n[:, 1] - a[1]) - ab[1] * (n[:, 0] - a[0])) / np.linalg.norm(ab)
    return p.iloc[int(d.argmax())]


def main():
    p = test_front(pd.read_csv(os.path.join(RES, "opt_pareto_test.csv")))
    r = pd.read_csv(os.path.join(RES, "opt_reference_test.csv"))
    p.to_csv(os.path.join(RES, "opt_pareto_test_front.csv"), index=False)
    rows = []
    for _, ref in r.iterrows():
        same_cost = p[p.test_cost_L <= ref.test_cost_L].sort_values("test_lost_pct").iloc[0]
        same_serv = p[p.test_lost_pct <= ref.test_lost_pct].sort_values("test_cost_L").iloc[0]
        rows.append(dict(reference=ref.policy, ref_cost_L=ref.test_cost_L,
                         ref_lost_pct=ref.test_lost_pct, ref_total_L=ref.cost_total / 1e5,
                         same_cost_solution=same_cost.policy,
                         same_cost_lost_pct=same_cost.test_lost_pct,
                         same_service_solution=same_serv.policy,
                         same_service_cost_L=same_serv.test_cost_L,
                         cost_saving_pct=100 * (1 - same_serv.test_cost_L / ref.test_cost_L)))
    k = pd.DataFrame(rows)
    k.to_csv(os.path.join(RES, "opt_key_findings.csv"), index=False)
    b = p.loc[p.cost_total.idxmin()]
    kn = knee(p)
    pd.set_option("display.width", 250)
    print(k.round(2).to_string())
    print(f"\nfront on test: {len(p)} solutions, dual sourcing chosen in {int(p.dual_sourcing.sum())}")
    print(f"min TOTAL cost (incl. lost sales): {b.policy} total {b.cost_total/1e5:.2f}L, "
          f"operating {b.test_cost_L:.2f}L, lost {b.test_lost_pct:.2f}%")
    print(f"knee point: {kn.policy} operating {kn.test_cost_L:.2f}L, lost {kn.test_lost_pct:.2f}%, total {kn.cost_total/1e5:.2f}L")
    cols = ["policy", "test_cost_L", "test_lost_pct", "store_s", "store_S", "dc_s", "dc_S",
            "fg_s", "fg_S", "rm_s", "rm_S", "dual_sourcing", "recovery_time"]
    print(p.loc[[0, len(p)//4, len(p)//2, p.cost_total.idxmin(), 3*len(p)//4, len(p)-1], cols].round(2).to_string())
    print("\nreferences:\n", r[cols].round(2).to_string())


if __name__ == "__main__":
    main()
