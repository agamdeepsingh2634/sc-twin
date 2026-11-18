"""Figures for the baseline study -> results/fig_*.png"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from twin import SupplyChainTwin, REFERENCE_POLICIES, sample_scenario
from twin.metrics import rolling_fill_rate

from twin.paths import results_dir
RES = results_dir(ROOT)
COL = {"Lean (JIT)": "#2a78d6", "Buffered": "#eb6834", "Lean + Dual sourcing": "#1baf7a",
       "Base-stock": "#eda100"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_fill_rate(s):
    scen = list(dict.fromkeys(s.scenario))
    pols = list(COL)
    fig, ax = plt.subplots(figsize=(9, 7))
    h = 0.2
    for i, p in enumerate(pols):
        d = s[s.policy == p].set_index("scenario").loc[scen]
        y = np.arange(len(scen)) + (i - 1.5) * (h + 0.015)
        ax.barh(y, d.fill_rate * 100, height=h, color=COL[p], label=p,
                xerr=d.fill_rate_ci95 * 100, error_kw=dict(ecolor=INK2, lw=0.8))
    ax.set_yticks(np.arange(len(scen)), scen)
    ax.invert_yaxis()
    ax.set_xlim(85, 100.5)
    ax.axvline(95, color=INK2, lw=1, ls="--")
    ax.text(95.1, -0.6, "95% target", color=INK2, fontsize=9)
    ax.set_xlabel("Customer fill rate (%) — mean of 200 runs, 95% CI")
    style(ax)
    ax.legend(frameon=False, loc="lower left", ncol=4, bbox_to_anchor=(-0.25, -0.16))
    ax.set_title("Fill rate by disruption scenario and inventory strategy",
                 loc="left", color=INK, fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig1_fill_rate.png"), dpi=170)


def fig_cost(s):
    scen = list(dict.fromkeys(s.scenario))
    fig, ax = plt.subplots(figsize=(9, 7))
    h = 0.2
    for i, p in enumerate(COL):
        d = s[s.policy == p].set_index("scenario").loc[scen]
        y = np.arange(len(scen)) + (i - 1.5) * (h + 0.015)
        ax.barh(y, d.cost_total / 1e5, height=h, color=COL[p], label=p,
                xerr=d.cost_total_ci95 / 1e5, error_kw=dict(ecolor=INK2, lw=0.8))
    ax.set_yticks(np.arange(len(scen)), scen)
    ax.invert_yaxis()
    ax.set_xlabel("Total cost incl. lost-sales penalty (₹ lakh, 10 months)")
    style(ax)
    ax.legend(frameon=False, loc="lower left", ncol=4, bbox_to_anchor=(-0.25, -0.16))
    ax.set_title("Total supply-chain cost by scenario and strategy",
                 loc="left", color=INK, fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig2_cost.png"), dpi=170)


def fig_timeline():
    """One supplier-outage replication: how each strategy rides the shock."""
    ev = sample_scenario("Supplier outage", np.random.default_rng(11))
    st, du = ev[0]["start"], ev[0]["duration"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axvspan(st, st + du, color="#e34948", alpha=0.08, lw=0)
    ax.text(st + 1, 4, f"Primary supplier down ({du} days)", color=INK2, fontsize=9)
    for p in REFERENCE_POLICIES:
        L = SupplyChainTwin(p, ev, seed=11).run()
        r = rolling_fill_rate(L["sales"], L["demand"]) * 100
        ax.plot(np.arange(len(r)), r, color=COL[p.name], lw=2, label=p.name)
    ax.set_xlim(st - 30, st + du + 80)
    ax.set_ylim(0, 103)
    ax.axhline(95, color=INK2, lw=1, ls="--")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("7-day fill rate (%)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("Resilience curve: one supplier-outage run", loc="left",
                 color=INK, fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig3_resilience_curve.png"), dpi=170)


if __name__ == "__main__":
    s = pd.read_csv(os.path.join(RES, "baseline_summary.csv"))
    fig_fill_rate(s)
    fig_cost(s)
    fig_timeline()
    print("figures saved")
