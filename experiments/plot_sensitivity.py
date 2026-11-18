"""Sensitivity figures -> results/fig7..fig9"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.plot_baseline import COL, INK, INK2, GRID, SURF
from experiments.plot_optimization import FRONT, clean

from twin.paths import results_dir
RES = results_dir(ROOT)
PCOL = {"Lean (JIT)": COL["Lean (JIT)"], "Buffered": COL["Buffered"],
        "Optimised (knee)": FRONT, "Optimised + dual": COL["Lean + Dual sourcing"]}
PMARK = {"Lean (JIT)": "s", "Buffered": "D", "Optimised (knee)": "o", "Optimised + dual": "^"}


def fig_cost_robustness():
    a = pd.read_csv(os.path.join(RES, "sens_costs.csv"))
    names = {"lost_sale_penalty": "Lost-sale penalty", "holding_cost": "Holding cost",
             "ordering_cost": "Ordering cost"}
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for (f, n), c, mk in zip(names.items(), cols, ["o", "s", "^"]):
        d = a[a.factor == f]
        ax.plot(d.multiplier, d.saving_vs_best_ref_pct, color=c, lw=2, marker=mk, ms=7, label=n)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xlabel("Cost parameter multiplier (1.0 = base case)")
    ax.set_ylabel("Saving vs best reference policy (%)")
    ax.set_ylim(0, 25)
    clean(ax)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("Optimised frontier beats every reference policy under all cost assumptions",
                 loc="left", color=INK, fontsize=11.5, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig7_cost_robustness.png"), dpi=170)


def fig_operational():
    b = pd.read_csv(os.path.join(RES, "sens_operational.csv"))
    factors = list(dict.fromkeys(b.factor))
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, f in zip(axes.flat, factors):
        d = b[b.factor == f]
        for p in PCOL:
            q = d[d.policy == p].sort_values("level")
            ax.plot(q.level, q.lost_pct, color=PCOL[p], lw=2, marker=PMARK[p], ms=6, label=p)
        ax.set_title(f, loc="left", color=INK, fontsize=10)
        ax.set_ylabel("Lost sales (%)", fontsize=9)
        clean(ax)
    axes.flat[-1].axis("off")
    h, l = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, l, frameon=False, loc="center", fontsize=10)
    fig.suptitle("Operational sensitivity: expected lost sales across 7 scenarios",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig8_operational_sensitivity.png"), dpi=170)


def fig_breakeven():
    c = pd.read_csv(os.path.join(RES, "sens_breakeven.csv"))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    labels = [f"{int(v*100)}%" for v in c.backup_capacity]
    vals = c.breakeven_premium_inr.clip(lower=0)
    bars = ax.bar(labels, vals, color=COL["Lean + Dual sourcing"], width=0.55)
    for b_, v in zip(bars, c.breakeven_premium_inr):
        ax.text(b_.get_x() + b_.get_width() / 2, max(v, 0) + 3,
                "never pays" if v <= 0 else f"₹{v:.0f}", ha="center", color=INK2, fontsize=9)
    ax.axhline(120, color=INK2, lw=1, ls="--")
    ax.text(3.3, 123, "current premium ₹120", color=INK2, fontsize=9, ha="right")
    ax.set_xlabel("Backup supplier capacity (% of mean demand)")
    ax.set_ylabel("Break-even premium (₹ per case)")
    ax.set_ylim(0, 140)
    clean(ax, "y")
    ax.set_title("Dual sourcing pays only if the backup premium is below this line",
                 loc="left", color=INK, fontsize=11.5, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig9_dual_breakeven.png"), dpi=170)


if __name__ == "__main__":
    fig_cost_robustness()
    fig_operational()
    fig_breakeven()
    print("saved")
