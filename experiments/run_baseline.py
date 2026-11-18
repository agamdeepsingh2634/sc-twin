"""
Baseline Monte Carlo study: 3 reference policies x 7 disruption scenarios.

Design
  * N replications per (scenario, policy)
  * Common random numbers: replication r uses the SAME demand series and the
    SAME sampled disruption for every policy, so differences between policies
    come from the policy, not from luck (variance reduction).
  * Outputs: results/baseline_runs.csv (one row per run)
             results/baseline_summary.csv (mean and 95% CI)
             results/baseline_paired_tests.csv (policy vs Lean, paired)

Usage:  python -m experiments.run_baseline --reps 200
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twin import SupplyChainTwin, REFERENCE_POLICIES, BASESTOCK, SCENARIOS, sample_scenario, summarize
from twin.disruptions import DisruptionState

from twin.paths import results_dir
OUT = results_dir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run(reps: int, base_seed: int = 2026):
    rows = []
    for si, scen in enumerate(SCENARIOS):
        for r in range(reps):
            seed = base_seed + 10_000 * si + r
            events = sample_scenario(scen, np.random.default_rng(seed + 7))
            start = DisruptionState(events).first_start
            for pol in REFERENCE_POLICIES + [BASESTOCK]:
                log = SupplyChainTwin(pol, events, seed=seed).run()
                m = summarize(log, start)
                m.update(scenario=scen, policy=pol.name, rep=r,
                         dis_start=start,
                         dis_duration=events[0]["duration"] if events else 0,
                         dis_magnitude=events[0]["magnitude"] if events else 0)
                rows.append(m)
    return pd.DataFrame(rows)


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t = time.time()
    df = run(a.reps)
    df.to_csv(os.path.join(OUT, "baseline_runs.csv"), index=False)

    metrics = ["fill_rate", "cost_total", "cost_operating", "cost_lost_sales",
               "cost_backup_premium", "avg_inventory", "recovery_time",
               "time_to_impact", "impact_depth", "resilience_loss",
               "lost_sales_disr", "bullwhip_factory"]
    g = df.groupby(["scenario", "policy"], sort=False)
    summ = g[metrics].mean()
    for m in ["fill_rate", "cost_total", "recovery_time", "resilience_loss"]:
        summ[m + "_ci95"] = g[m].apply(ci95)
    summ["share_never_impacted"] = g["time_to_impact"].apply(lambda s: s.isna().mean())
    summ.reset_index().to_csv(os.path.join(OUT, "baseline_summary.csv"), index=False)

    # paired tests vs Lean (Wilcoxon signed-rank on total cost and fill rate)
    tests = []
    for scen in SCENARIOS:
        base = df[(df.scenario == scen) & (df.policy == "Lean (JIT)")].set_index("rep")
        for pol in df.policy.unique():
            if pol == "Lean (JIT)":
                continue
            other = df[(df.scenario == scen) & (df.policy == pol)].set_index("rep")
            for m in ["cost_total", "fill_rate"]:
                diff = (other[m] - base[m]).values
                p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                tests.append(dict(scenario=scen, policy=pol, metric=m,
                                  mean_diff_vs_lean=diff.mean(), p_value=p))
    pd.DataFrame(tests).to_csv(os.path.join(OUT, "baseline_paired_tests.csv"), index=False)
    print(f"{len(df)} runs in {time.time() - t:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
