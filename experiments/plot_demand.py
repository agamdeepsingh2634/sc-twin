"""Figure: real M5 demand vs the synthetic generator -> results/m5/fig13_demand_real_vs_synthetic.png"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from twin import m5, NetworkConfig
from twin.demand import generate_store_demand
from experiments.plot_baseline import INK, INK2, GRID
from experiments.plot_optimization import clean

NPZ = os.path.join(ROOT, "results", "m5_foods_daily.npz")
OUT = os.path.join(ROOT, "results", "m5")

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    m5.enable(NPZ)
    net = NetworkConfig()
    seed = 12
    real = m5.demand_for(seed, net.store_base_demand)
    syn = generate_store_demand(net.store_base_demand, 365, np.random.default_rng(seed))
    d0, d1 = m5.window_dates(seed)
    tot_r = sum(real.values()); tot_s = sum(syn.values())
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[2.2, 1]))
    a1.plot(tot_s, color="#9ca3af", lw=1.1, label="Synthetic generator")
    a1.plot(tot_r, color="#2a78d6", lw=1.4, label=f"Real Walmart M5 (FOODS, {d0} to {d1})")
    a1.set_xlabel("Day of simulation year"); a1.set_ylabel("Total demand, cases/day")
    a1.legend(frameon=False, loc="upper left", fontsize=9); clean(a1, "y")
    a1.set_title("Total daily demand over one simulated year", loc="left", color=INK, fontsize=11)
    # weekly profile of the full real series vs synthetic
    z = np.load(NPZ); stores = list(z["stores"]); dates = z["dates"]
    import pandas as pd
    wd = pd.to_datetime(dates).dayofweek.values
    real_all = sum(z["sales"][stores.index(r)] for r in m5.STORE_MAP.values())
    prof_r = np.array([real_all[wd == k].mean() for k in range(7)]); prof_r /= prof_r.mean()
    from twin.demand import WEEKLY
    prof_s = WEEKLY / WEEKLY.mean()
    x = np.arange(7); w = 0.38
    a2.bar(x - w / 2, prof_s, w, color="#9ca3af", label="Synthetic")
    a2.bar(x + w / 2, prof_r, w, color="#2a78d6", label="Real M5")
    a2.set_xticks(x, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    a2.axhline(1, color=INK2, lw=0.8)
    a2.set_ylim(0.7, 1.3); a2.set_ylabel("Relative to weekly mean")
    a2.legend(frameon=False, fontsize=9); clean(a2, "y")
    a2.set_title("Day-of-week pattern", loc="left", color=INK, fontsize=11)
    cv_r = np.std(tot_r) / np.mean(tot_r); cv_s = np.std(tot_s) / np.mean(tot_s)
    fig.suptitle(f"Real vs synthetic demand (coefficient of variation: real {cv_r:.2f}, synthetic {cv_s:.2f})",
                 x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig13_demand_real_vs_synthetic.png"), dpi=170)
    print("saved", cv_r, cv_s)
