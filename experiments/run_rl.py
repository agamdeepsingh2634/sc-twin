"""
Train the Q-learning sourcing agent and compare it with the optimised policy.

Training: N episodes, each a full year with a scenario drawn at random from the
7-scenario library (fresh seeds every episode). Epsilon decays 0.30 -> 0.02.
Testing : greedy agent vs Optimised (knee) and Optimised + dual on the SAME 210
held-out runs used to test the Pareto frontier (seed base 900000).

Outputs: results/rl_training.csv, rl_test.csv, rl_policy.csv, fig11_rl.png
"""
import os, sys, time, pickle
from dataclasses import replace
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from twin import SupplyChainTwin, SCENARIOS, sample_scenario, summarize
from twin.disruptions import DisruptionState
from twin.optimize import make_cases
from twin.rl_agent import QSourcingAgent, ACTION_NAMES
from experiments.run_sensitivity import knee_policy
from experiments.plot_baseline import COL, INK, INK2
from experiments.plot_optimization import FRONT, clean

from twin.paths import results_dir
RES = results_dir(ROOT)
SCEN = list(SCENARIOS)


def train(episodes=4000, seed=0):
    kp, _ = knee_policy()
    agent = QSourcingAgent(seed=seed)
    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(episodes):
        agent.epsilon = max(0.02, 0.30 * (1 - ep / (0.8 * episodes)))
        agent.reset()
        s = int(rng.integers(1e9))
        ev = sample_scenario(SCEN[ep % len(SCEN)], np.random.default_rng(s))
        log = SupplyChainTwin(kp, ev, seed=s, rm_agent=agent).run()
        m = summarize(log, DisruptionState(ev).first_start)
        hist.append(dict(episode=ep + 1, cost_total_L=m["cost_total"] / 1e5,
                         fill_rate=m["fill_rate"], epsilon=agent.epsilon))
    return agent, pd.DataFrame(hist)


def evaluate(agent, reps=30):
    kp, kd = knee_policy()
    agent.learning = False
    rows = []
    for seed, ev in make_cases(reps, base_seed=900_000):
        scen = ev[0]["label"] if ev else "No disruption"
        start = DisruptionState(ev).first_start
        for name, pol, ag in [("Optimised (knee)", kp, None),
                              ("Optimised + dual", kd, None),
                              ("Q-learning agent", kp, agent)]:
            if ag:
                ag.reset()
            m = summarize(SupplyChainTwin(pol, ev, seed=seed, rm_agent=ag).run(), start)
            m.update(policy=name, scenario=scen)
            if ag:
                for i, n in enumerate(ACTION_NAMES):
                    m[f"act_{n}"] = ag.action_counts[i]
            rows.append(m)
    return pd.DataFrame(rows)


def learned_policy(agent):
    """Readable table: greedy action in each state."""
    rows = []
    it = np.ndindex(agent.Q.shape[:-1])
    for st in it:
        q = agent.Q[st]
        if np.all(q == 0):
            continue                      # never visited
        rows.append(dict(rm_ip_bin=st[0], fg_bin=st[1], s1_disrupted=st[2],
                         stuck_bin=st[3], action=ACTION_NAMES[int(q.argmax())]))
    return pd.DataFrame(rows)


def figure(hist, test):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw=dict(width_ratios=[1.2, 1]))
    roll = hist.cost_total_L.rolling(200, min_periods=50).mean()
    a1.plot(hist.episode, roll, color=FRONT, lw=2)
    a1.set_xlabel("Training episode (one simulated year each)")
    a1.set_ylabel("Total cost, ₹ lakh (200-episode average)")
    a1.set_title("Learning curve", loc="left", color=INK, fontsize=11)
    clean(a1)
    g = test.groupby("policy", sort=False).agg(cost=("cost_total", "mean"),
                                               lost=("fill_rate", lambda x: 100 * (1 - x.mean())))
    cols = [FRONT, COL["Lean + Dual sourcing"], COL["Base-stock"]]
    bars = a2.bar(g.index, g.cost / 1e5, color=cols, width=0.55)
    for b, (c, l) in zip(bars, g.values):
        a2.text(b.get_x() + b.get_width() / 2, c / 1e5 + 0.2, f"₹{c/1e5:.1f}L\n{l:.2f}% lost",
                ha="center", color=INK2, fontsize=9)
    a2.set_ylabel("Total cost incl. lost sales (₹ lakh)")
    a2.set_ylim(0, g.cost.max() / 1e5 * 1.25)
    a2.set_title("Held-out test (210 runs)", loc="left", color=INK, fontsize=11)
    a2.tick_params(axis="x", labelsize=9)
    clean(a2, "y")
    fig.suptitle("Reinforcement-learning sourcing agent", x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "fig11_rl.png"), dpi=170)


if __name__ == "__main__":
    t = time.time()
    agent, hist = train(int(sys.argv[1]) if len(sys.argv) > 1 else 4000)
    print(f"trained in {time.time() - t:.0f}s")
    hist.to_csv(os.path.join(RES, "rl_training.csv"), index=False)
    with open(os.path.join(RES, "rl_qtable.pkl"), "wb") as f:
        pickle.dump(agent.Q, f)
    test = evaluate(agent)
    test.to_csv(os.path.join(RES, "rl_test.csv"), index=False)
    learned_policy(agent).to_csv(os.path.join(RES, "rl_policy.csv"), index=False)
    figure(hist, test)
    pd.set_option("display.width", 200)
    print(test.groupby(["scenario", "policy"], sort=False).agg(
        total_L=("cost_total", lambda x: x.mean() / 1e5),
        lost=("fill_rate", lambda x: 100 * (1 - x.mean()))).round(2).unstack().to_string())
    print(test.groupby("policy", sort=False).agg(total_L=("cost_total", lambda x: x.mean() / 1e5),
          op_L=("cost_operating", lambda x: x.mean() / 1e5),
          lost=("fill_rate", lambda x: 100 * (1 - x.mean())),
          rec=("recovery_time", "mean")).round(2))
    acts = [c for c in test.columns if c.startswith("act_")]
    print(test[test.policy == "Q-learning agent"][acts].sum() / test[test.policy == "Q-learning agent"][acts].sum().sum())
    print(f"total {time.time() - t:.0f}s")
