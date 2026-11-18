"""
Sensitivity analysis.

Part A - cost parameters (post-hoc, exact): with a policy fixed, the simulation
  does not depend on cost parameters, so total cost under new prices is a
  re-weighting of the stored cost components. We re-price every solution on the
  test frontier and the three reference policies, and ask: which solution is
  now cheapest, and how much does it save vs the best reference?

Part B - operational parameters (re-simulated, 7 scenarios x 20 runs, CRN):
  S1 lead time, factory capacity, disruption severity, demand noise, and
  backup-supplier capacity. Policies: Lean, Buffered, Knee (optimised),
  Knee + dual sourcing.

Part C - dual-sourcing break-even: for each backup capacity, the backup premium
  (INR/case) at which Knee+Dual costs the same as Knee.

Outputs: results/sens_costs.csv, sens_operational.csv, sens_breakeven.csv
"""
import os, sys, time
from dataclasses import replace
from multiprocessing import Pool
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from twin.config import NetworkConfig, CostConfig, Policy, LEAN, BUFFERED
from twin.optimize import make_cases, evaluate_policy
from experiments.analyze_optimization import test_front, knee

from twin.paths import results_dir
RES = results_dir(ROOT)
REPS = 20
BASE = CostConfig()


def knee_policy():
    k = knee(test_front(pd.read_csv(os.path.join(RES, "opt_pareto_test.csv"))))
    pol = Policy("Optimised (knee)", store=(k.store_s, k.store_S), dc=(k.dc_s, k.dc_S),
                 fg=(k.fg_s, k.fg_S), rm=(k.rm_s, k.rm_S), dual_sourcing=False)
    return pol, replace(pol, name="Optimised + dual", dual_sourcing=True)


# ---------------------------------------------------------------- Part A
def part_a():
    p = test_front(pd.read_csv(os.path.join(RES, "opt_pareto_test.csv")))
    r = pd.read_csv(os.path.join(RES, "opt_reference_test.csv"))
    factors = {"lost_sale_penalty": [0.4, 0.7, 1.0, 1.5, 2.0],
               "holding_cost": [0.5, 0.75, 1.0, 1.5, 2.0],
               "ordering_cost": [0.25, 0.5, 1.0, 1.5, 2.0]}
    col = {"lost_sale_penalty": "cost_lost_sales", "holding_cost": "cost_holding",
           "ordering_cost": "cost_ordering"}
    rows = []
    for f, levels in factors.items():
        for m in levels:
            def total(df):
                t = df.cost_holding + df.cost_ordering + df.cost_lost_sales + df.cost_backup_premium
                return (t + (m - 1) * df[col[f]]) / 1e5
            pt, rt = total(p), total(r)
            best = p.loc[pt.idxmin()]
            row = dict(factor=f, multiplier=m, best_solution=best.policy,
                       best_total_L=pt.min(), best_lost_pct=best.test_lost_pct,
                       best_store_S=best.store_S, best_dc_S=best.dc_S,
                       best_fg_S=best.fg_S, best_rm_S=best.rm_S)
            for i, ref in r.iterrows():
                row[f"{ref.policy}_total_L"] = rt[i]
            row["best_reference_total_L"] = rt.min()
            row["saving_vs_best_ref_pct"] = 100 * (1 - pt.min() / rt.min())
            rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "sens_costs.csv"), index=False)


# ---------------------------------------------------------------- Part B
def job(args):
    factor, level, pol, net, severity = args
    m = evaluate_policy(pol, make_cases(REPS, base_seed=500_000, severity=severity), net=net)
    m.update(factor=factor, level=level, policy=pol.name,
             s2_units=m["cost_backup_premium"] / BASE.backup_premium)
    return m


def part_b():
    kp, kd = knee_policy()
    pols = [LEAN, BUFFERED, kp, kd]
    base = NetworkConfig()
    grid = {
        "S1 lead time (days)": [(v, replace(base, lt_s1_to_factory=v), 1.0) for v in (5, 10, 15, 20)],
        "Factory capacity (x demand)": [(v, replace(base, factory_capacity_mult=v), 1.0) for v in (1.10, 1.25, 1.50)],
        "Disruption severity (x duration)": [(v, base, v) for v in (0.5, 1.0, 1.5, 2.0)],
        "Demand noise (NB dispersion)": [(v, replace(base, demand_dispersion=v), 1.0) for v in (5, 25, 100)],
        "Backup capacity (x demand)": [(v, replace(base, s2_capacity_mult=v), 1.0) for v in (0.4, 0.6, 0.8, 1.0)],
    }
    jobs = [(f, lv, p, net, sev) for f, levels in grid.items() for lv, net, sev in levels for p in pols]
    with Pool(os.cpu_count()) as pool:
        out = pool.map(job, jobs)
    df = pd.DataFrame(out)
    df["op_cost_L"] = df.cost_operating / 1e5
    df["total_cost_L"] = df.cost_total / 1e5
    df["lost_pct"] = 100 * (1 - df.fill_rate)
    df.to_csv(os.path.join(RES, "sens_operational.csv"), index=False)
    return df


# ---------------------------------------------------------------- Part C
def part_c(df):
    b = df[df.factor == "Backup capacity (x demand)"]
    rows = []
    for lv in sorted(b.level.unique()):
        k = b[(b.level == lv) & (b.policy == "Optimised (knee)")].iloc[0]
        d = b[(b.level == lv) & (b.policy == "Optimised + dual")].iloc[0]
        # total cost of dual excluding premium, as a function of premium P:
        #   dual_total(P) = (d.cost_total - d.premium_paid) + P * s2_units
        base_dual = d.cost_total - d.cost_backup_premium
        gain = k.cost_total - base_dual          # benefit before paying premium
        be = gain / d.s2_units if d.s2_units > 0 else np.nan
        rows.append(dict(backup_capacity=lv, knee_total_L=k.cost_total / 1e5,
                         dual_total_at_120_L=d.cost_total / 1e5,
                         knee_lost_pct=100 * (1 - k.fill_rate),
                         dual_lost_pct=100 * (1 - d.fill_rate),
                         s2_cases_per_run=d.s2_units,
                         breakeven_premium_inr=be))
    pd.DataFrame(rows).to_csv(os.path.join(RES, "sens_breakeven.csv"), index=False)


if __name__ == "__main__":
    t = time.time()
    part_a()
    df = part_b()
    part_c(df)
    print(f"done in {time.time() - t:.0f}s")
