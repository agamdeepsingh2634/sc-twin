"""Figures for the optimisation study -> results/fig4..fig6"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.analyze_optimization import test_front, knee
from experiments.plot_baseline import COL, INK, INK2, GRID, SURF

from twin.paths import results_dir
RES = results_dir(ROOT)
FRONT = "#6250d6"


def clean(ax, grid="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_pareto(p, r, k):
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.plot(p.test_cost_L, p.test_lost_pct, color=FRONT, lw=2, zorder=2)
    ax.scatter(p.test_cost_L, p.test_lost_pct, s=14, color=FRONT, zorder=3,
               label="Optimised frontier (NSGA-II)")
    marks = {"Lean (JIT)": "s", "Buffered": "D", "Lean + Dual sourcing": "^"}
    for _, row in r.iterrows():
        ax.scatter(row.test_cost_L, row.test_lost_pct, s=90, marker=marks[row.policy],
                   color=COL[row.policy], edgecolor=SURF, lw=2, zorder=4)
        ax.annotate(row.policy, (row.test_cost_L, row.test_lost_pct), xytext=(8, 6),
                    textcoords="offset points", color=INK, fontsize=9)
    ax.scatter(k.test_cost_L, k.test_lost_pct, s=160, facecolor="none",
               edgecolor=INK, lw=1.5, zorder=5)
    ax.annotate(f"Knee point\n₹{k.test_cost_L:.1f}L, {k.test_lost_pct:.1f}% lost",
                (k.test_cost_L, k.test_lost_pct), xytext=(-130, -55),
                textcoords="offset points", color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    lean = r[r.policy == "Lean (JIT)"].iloc[0]
    kf = pd.read_csv(os.path.join(RES, "opt_key_findings.csv")).set_index("reference").loc["Lean (JIT)"]
    ax.annotate("", xy=(kf.same_service_cost_L, lean.test_lost_pct), xytext=(lean.test_cost_L - 0.15, lean.test_lost_pct),
                arrowprops=dict(arrowstyle="->", color=INK2, lw=1))
    ax.text(kf.same_service_cost_L + 0.4, lean.test_lost_pct - 0.45,
            f"same service, {kf.cost_saving_pct:.0f}% cheaper", color=INK2, fontsize=9)
    ax.set_xlabel("Expected operating cost — holding + ordering + backup premium (₹ lakh)")
    ax.set_ylabel("Expected lost sales (% of demand)")
    ax.set_ylim(-0.3, 10.5)
    clean(ax)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("The price of resilience: cost vs lost sales across 7 disruption scenarios",
                 loc="left", color=INK, fontsize=12, pad=10)
    data = "real Walmart M5 demand" if os.environ.get("SCTWIN_M5") else "synthetic demand"
    fig.text(0.01, 0.01, f"Out-of-sample test: 210 fresh simulation runs per point, {data}",
             color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(os.path.join(RES, "fig4_pareto_frontier.png"), dpi=170)


def fig_convergence(h):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(h.gen, h.hypervolume, color=FRONT, lw=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    clean(ax, "y")
    ax.set_title("NSGA-II convergence (higher = better frontier)", loc="left",
                 color=INK, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig5_convergence.png"), dpi=170)


def fig_where_stock(r, k):
    ech = [("Stores", "store"), ("DCs", "dc"), ("Factory FG", "fg"), ("Raw material", "rm")]
    rows = [("Lean (JIT)", r[r.policy == "Lean (JIT)"].iloc[0], COL["Lean (JIT)"]),
            ("Buffered", r[r.policy == "Buffered"].iloc[0], COL["Buffered"]),
            ("Optimised (knee)", k, FRONT)]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    w = 0.26
    x = np.arange(len(ech))
    for i, (name, row, c) in enumerate(rows):
        vals = [row[f"{e}_S"] for _, e in ech]
        b = ax.bar(x + (i - 1) * (w + 0.02), vals, width=w, color=c, label=name)
        ax.bar_label(b, fmt="%.0f", fontsize=8, color=INK2, padding=2)
    ax.set_xticks(x, [e for e, _ in ech])
    ax.set_ylabel("Order-up-to level S (days of cover)")
    clean(ax, "y")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Where the optimiser puts inventory", loc="left", color=INK,
                 fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig6_where_stock.png"), dpi=170)


if __name__ == "__main__":
    p = test_front(pd.read_csv(os.path.join(RES, "opt_pareto_test.csv")))
    r = pd.read_csv(os.path.join(RES, "opt_reference_test.csv"))
    k = knee(p)
    fig_pareto(p, r, k)
    fig_convergence(pd.read_csv(os.path.join(RES, "opt_history.csv")))
    fig_where_stock(r, k)
    print("saved")
