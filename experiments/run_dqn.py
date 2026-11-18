"""
Train the NumPy Double-DQN sourcing agent (CPU) and compare it with the
optimised (knee) policy and the tabular Q-learning agent on the same 210
held-out runs used everywhere else.

Usage:
    python -m experiments.run_dqn --episodes 1500            # synthetic demand
    SCTWIN_M5=results/m5_foods_daily.npz python -m experiments.run_dqn   # real M5

Outputs (suffix _m5 when real data is on):
    results/dqn_training*.csv, dqn_test*.csv, dqn_weights*.npz, fig12_dqn*.png
"""
import argparse, os, sys, time, pickle
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from twin import SupplyChainTwin, SCENARIOS, sample_scenario, summarize, m5
from twin.disruptions import DisruptionState
from twin.optimize import make_cases
from twin.dqn import DQNAgent
from twin.rl_agent import QSourcingAgent, ACTION_NAMES
from experiments.run_sensitivity import knee_policy
from experiments.plot_baseline import COL, INK, INK2
from experiments.plot_optimization import FRONT, clean

from twin.paths import results_dir
RES = results_dir(ROOT)
SCEN = list(SCENARIOS)


def train(episodes, seed=0):
    kp, _ = knee_policy()
    agent = DQNAgent(seed=seed)
    rng = np.random.default_rng(seed + 1)
    hist, t0 = [], time.time()
    for ep in range(episodes):
        agent.epsilon = max(0.05, 1.0 - ep / (0.6 * episodes))
        agent.reset()
        s = int(rng.integers(1e9))
        ev = sample_scenario(SCEN[ep % len(SCEN)], np.random.default_rng(s))
        log = SupplyChainTwin(kp, ev, seed=s, rm_agent=agent).run()
        agent.end_episode()
        m = summarize(log, DisruptionState(ev).first_start)
        hist.append(dict(episode=ep + 1, cost_total_L=m["cost_total"] / 1e5,
                         fill_rate=m["fill_rate"], epsilon=agent.epsilon,
                         loss=np.mean(agent.losses[-500:]) if agent.losses else np.nan))
        if (ep + 1) % max(1, episodes // 10) == 0:
            h = pd.DataFrame(hist[-100:])
            print(f"ep {ep+1:5d}  eps={agent.epsilon:.2f}  cost(100-ep avg)={h.cost_total_L.mean():.2f}L  "
                  f"fill={h.fill_rate.mean()*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    return agent, pd.DataFrame(hist)


def evaluate(dqn, reps=30):
    kp, _ = knee_policy()
    q = QSourcingAgent()
    with open(os.path.join(ROOT, "results", "rl_qtable.pkl"), "rb") as f:
        q.Q = pickle.load(f)
    q.learning = False
    dqn.learning = False
    rows = []
    for seed, ev in make_cases(reps, base_seed=900_000):
        scen = ev[0]["label"] if ev else "No disruption"
        st = DisruptionState(ev).first_start
        for name, ag in [("Optimised (knee)", None), ("Q-learning (tabular)", q), ("Deep Q-network", dqn)]:
            if ag:
                ag.reset()
            m = summarize(SupplyChainTwin(kp, ev, seed=seed, rm_agent=ag).run(), st)
            m.update(policy=name, scenario=scen)
            if ag is dqn:
                for i, n in enumerate(ACTION_NAMES):
                    m[f"act_{n}"] = ag.action_counts[i]
            rows.append(m)
    return pd.DataFrame(rows)


def figure(hist, test, suffix):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    a1.plot(hist.episode, hist.cost_total_L.rolling(100, min_periods=20).mean(), color=FRONT, lw=2)
    a1.set_xlabel("Training episode (one simulated year each)")
    a1.set_ylabel("Total cost, ₹ lakh (100-episode average)")
    a1.set_title("DQN learning curve", loc="left", color=INK, fontsize=11)
    clean(a1)
    g = test.groupby("policy", sort=False).agg(cost=("cost_total", "mean"),
                                               lost=("fill_rate", lambda x: 100 * (1 - x.mean())))
    bars = a2.bar(g.index, g.cost / 1e5, color=[FRONT, COL["Base-stock"], COL["Lean + Dual sourcing"]], width=0.55)
    for b, (c, l) in zip(bars, g.values):
        a2.text(b.get_x() + b.get_width() / 2, c / 1e5 + 0.2, f"₹{c/1e5:.1f}L\n{l:.2f}% lost",
                ha="center", color=INK2, fontsize=9)
    a2.set_ylim(0, g.cost.max() / 1e5 * 1.25)
    a2.set_ylabel("Total cost incl. lost sales (₹ lakh)")
    a2.set_title("Held-out test (210 runs)", loc="left", color=INK, fontsize=11)
    a2.tick_params(axis="x", labelsize=9)
    clean(a2, "y")
    data = "real Walmart M5 demand" if suffix else "synthetic demand"
    fig.suptitle(f"Deep Q-network sourcing agent ({data})", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f"fig12_dqn{suffix}.png"), dpi=170)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    a = ap.parse_args()
    suffix = "_m5" if m5.enabled() else ""
    t = time.time()
    agent, hist = train(a.episodes)
    hist.to_csv(os.path.join(RES, f"dqn_training{suffix}.csv"), index=False)
    agent.save(os.path.join(RES, f"dqn_weights{suffix}.npz"))
    test = evaluate(agent)
    test.to_csv(os.path.join(RES, f"dqn_test{suffix}.csv"), index=False)
    figure(hist, test, suffix)
    pd.set_option("display.width", 200)
    print(test.groupby("policy", sort=False).agg(
        total_L=("cost_total", lambda x: x.mean() / 1e5), op_L=("cost_operating", lambda x: x.mean() / 1e5),
        lost=("fill_rate", lambda x: 100 * (1 - x.mean())), rec=("recovery_time", "mean")).round(2))
    acts = [c for c in test.columns if c.startswith("act_")]
    d = test[test.policy == "Deep Q-network"][acts].sum()
    print((d / d.sum()).round(3))
    print(f"total {time.time() - t:.0f}s")
