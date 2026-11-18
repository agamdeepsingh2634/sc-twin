"""
Extra analyses
  1. Two-way ANOVA (scenario x policy) on the baseline runs, with effect sizes
     (eta-squared) -> which factor explains most of the variation?
  2. Single-point-of-failure (node criticality): each node is shut for 21 days
     at a random date; rank nodes by lost sales, time-to-impact and recovery.

Outputs: results/anova.csv, results/node_criticality.csv, fig10_node_criticality.png
"""
import os, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from twin import SupplyChainTwin, LEAN, BUFFERED, summarize
from experiments.run_sensitivity import knee_policy
from experiments.plot_baseline import COL, INK, INK2
from experiments.plot_optimization import FRONT, clean

from twin.paths import results_dir
RES = results_dir(ROOT)


# ------------------------------------------------------------------ ANOVA
def two_way_anova(df, y, a="scenario", b="policy"):
    """Balanced two-way ANOVA with interaction (Type I = Type III when balanced)."""
    g = df[y].mean()
    la, lb = df[a].unique(), df[b].unique()
    n = len(df) / (len(la) * len(lb))
    ma = df.groupby(a)[y].mean()
    mb = df.groupby(b)[y].mean()
    mab = df.groupby([a, b])[y].mean()
    ss_a = n * len(lb) * ((ma - g) ** 2).sum()
    ss_b = n * len(la) * ((mb - g) ** 2).sum()
    cell = df.set_index([a, b]).index.map(mab)
    ss_ab = n * sum((mab[(i, j)] - ma[i] - mb[j] + g) ** 2 for i in la for j in lb)
    ss_e = ((df[y].values - np.asarray(cell)) ** 2).sum()
    ss_t = ((df[y] - g) ** 2).sum()
    dfa, dfb = len(la) - 1, len(lb) - 1
    dfab, dfe = dfa * dfb, len(df) - len(la) * len(lb)
    rows = []
    for name, ss, d in [(a, ss_a, dfa), (b, ss_b, dfb), (f"{a} x {b}", ss_ab, dfab)]:
        F = (ss / d) / (ss_e / dfe)
        rows.append(dict(response=y, source=name, SS=ss, df=d, F=F,
                         p_value=stats.f.sf(F, d, dfe), eta_sq=ss / ss_t))
    rows.append(dict(response=y, source="error", SS=ss_e, df=dfe, F=np.nan,
                     p_value=np.nan, eta_sq=ss_e / ss_t))
    return rows


def anova():
    df = pd.read_csv(os.path.join(RES, "baseline_runs.csv"))
    df["cost_total_L"] = df.cost_total / 1e5
    rows = two_way_anova(df, "fill_rate") + two_way_anova(df, "cost_total_L")
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RES, "anova.csv"), index=False)
    return out


# ------------------------------------------------------- node criticality
NODES = {
    "Supplier S1": dict(type="supplier_outage"),
    "Port / S1 lane": dict(type="port_blockage", magnitude=21),
    "Factory": dict(type="factory_capacity", magnitude=0.0),
    "DC North": dict(type="dc_outage", target="DC_North"),
    "DC West": dict(type="dc_outage", target="DC_West"),
}


def crit_job(args):
    node, pol, r = args
    rng = np.random.default_rng(70_000 + r)
    start = int(rng.integers(100, 200))
    spec = NODES[node]
    ev = [dict(type=spec["type"], target=spec.get("target"), start=start, duration=21,
               magnitude=spec.get("magnitude", 0.0), label=node)]
    log = SupplyChainTwin(pol, ev, seed=70_000 + r).run()
    m = summarize(log, start)
    m.update(node=node, policy=pol.name, rep=r)
    return m


def criticality(reps=100):
    kp, _ = knee_policy()
    jobs = [(n, p, r) for n in NODES for p in (LEAN, BUFFERED, kp) for r in range(reps)]
    with Pool(os.cpu_count()) as pool:
        df = pd.DataFrame(pool.map(crit_job, jobs))
    s = df.groupby(["node", "policy"], sort=False).agg(
        lost_sales_cases=("lost_sales_disr", "mean"),
        lost_pct=("fill_rate", lambda x: 100 * (1 - x.mean())),
        time_to_impact=("time_to_impact", "mean"),
        share_never_impacted=("time_to_impact", lambda x: x.isna().mean()),
        recovery_time=("recovery_time", "mean")).reset_index()
    s.to_csv(os.path.join(RES, "node_criticality.csv"), index=False)
    return s


def fig_criticality(s):
    order = (s[s.policy == "Lean (JIT)"].sort_values("lost_sales_cases", ascending=False).node.tolist())
    pcol = {"Lean (JIT)": COL["Lean (JIT)"], "Buffered": COL["Buffered"], "Optimised (knee)": FRONT}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    h = 0.26
    for i, (p, c) in enumerate(pcol.items()):
        d = s[s.policy == p].set_index("node").loc[order]
        y = np.arange(len(order)) + (i - 1) * (h + 0.02)
        ax.barh(y, d.lost_sales_cases, height=h, color=c, label=p)
    ax.set_yticks(np.arange(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("Cases lost in the 120 days after a 21-day node shutdown (mean of 100 runs)")
    clean(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Single point of failure: which node hurts most when it goes down?",
                 loc="left", color=INK, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig10_node_criticality.png"), dpi=170)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(anova().round(4).to_string())
    s = criticality()
    print(s.round(2).to_string())
    fig_criticality(s)
