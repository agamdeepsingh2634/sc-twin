"""
Tabular Q-learning tuning with best-checkpoint selection (CPU).

Same protocol as experiments/tune_dqn_torch.py so the two RL agents are
treated identically:
  * several settings (learning rate alpha, discount gamma, exploration) train
    in parallel, one process each
  * every VAL_EVERY episodes the greedy agent is scored on the fixed
    validation set (35 runs, seed base 700000, never used for testing) and the
    best Q-table is kept
  * the best Q-table of each setting is scored on the same 210 held-out test
    runs (seed base 900000) as every other method

Usage:  python -m experiments.tune_qlearning [--quick]
        (set SCTWIN_M5=results\\m5_foods_daily.npz first for real demand)
Outputs: qlearning_tuning_summary.csv, qlearning_tuning_curves.csv,
         qlearning_tuned_<cfg>.pkl  (in results/ or results/m5/)
"""
import argparse, os, pickle, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CONFIGS = {
    # name: (alpha, gamma, eps_start, eps_end, explore_frac)
    "Q1_base_a0.10_g0.97":    (0.10, 0.97, 0.30, 0.02, 0.8),
    "Q2_slow_a0.05_g0.97":    (0.05, 0.97, 0.30, 0.02, 0.8),
    "Q3_fast_a0.20_g0.97":    (0.20, 0.97, 0.30, 0.02, 0.8),
    "Q4_farsight_a0.10_g0.99": (0.10, 0.99, 0.30, 0.02, 0.8),
    "Q5_shortsight_a0.10_g0.93": (0.10, 0.93, 0.30, 0.02, 0.8),
    "Q6_explore_a0.10_g0.97":  (0.10, 0.97, 0.60, 0.02, 0.9),
}


def run_config(args):
    name, cfg, episodes, val_every, test_reps, m5_path = args
    if m5_path:
        os.environ["SCTWIN_M5"] = m5_path
    from twin import SupplyChainTwin, SCENARIOS, sample_scenario, summarize
    from twin.disruptions import DisruptionState
    from twin.optimize import make_cases
    from twin.paths import results_dir
    from twin.rl_agent import QSourcingAgent
    from experiments.run_sensitivity import knee_policy

    alpha, gamma, e0, e1, explore = cfg
    res = results_dir(ROOT)
    kp, _ = knee_policy()
    scen = list(SCENARIOS)
    agent = QSourcingAgent(alpha=alpha, gamma=gamma, seed=0)
    val_cases = make_cases(1 if episodes < 1000 else 5, base_seed=700_000)

    def score(cases):
        agent.learning = False
        rows = []
        for seed, ev in cases:
            agent.reset()
            rows.append(summarize(SupplyChainTwin(kp, ev, seed=seed, rm_agent=agent).run(),
                                  DisruptionState(ev).first_start))
        agent.learning = True
        return rows

    rng = np.random.default_rng(0)
    best_val, best_Q, best_ep, curve = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(episodes):
        agent.epsilon = max(e1, e0 * (1 - ep / (explore * episodes)))
        agent.reset()
        s = int(rng.integers(1e9))
        ev = sample_scenario(scen[ep % len(scen)], np.random.default_rng(s))
        SupplyChainTwin(kp, ev, seed=s, rm_agent=agent).run()
        if (ep + 1) % val_every == 0:
            rows = score(val_cases)
            v = float(np.mean([r["cost_total"] for r in rows])) / 1e5
            lost = 100 * (1 - float(np.mean([r["fill_rate"] for r in rows])))
            curve.append(dict(config=name, episode=ep + 1, val_cost_L=v, val_lost_pct=lost))
            if v < best_val:
                best_val, best_ep, best_Q = v, ep + 1, agent.Q.copy()
            print(f"  [{name}] ep {ep+1:6d} val={v:6.2f}L lost={lost:5.2f}% "
                  f"best={best_val:6.2f}L@{best_ep} ({time.time()-t0:.0f}s)", flush=True)

    agent.Q = best_Q
    with open(os.path.join(res, f"qlearning_tuned_{name}.pkl"), "wb") as f:
        pickle.dump(best_Q, f)
    test = score(make_cases(test_reps, base_seed=900_000))
    rec = [r["recovery_time"] for r in test if r["recovery_time"] == r["recovery_time"]]
    return dict(config=name, alpha=alpha, gamma=gamma, eps_start=e0, eps_end=e1,
                explore_frac=explore, episodes=episodes, best_episode=best_ep,
                best_val_cost_L=best_val,
                test_total_L=np.mean([r["cost_total"] for r in test]) / 1e5,
                test_operating_L=np.mean([r["cost_operating"] for r in test]) / 1e5,
                test_lost_pct=100 * (1 - np.mean([r["fill_rate"] for r in test])),
                test_recovery=float(np.mean(rec)) if rec else 0.0,
                minutes=(time.time() - t0) / 60), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=14000)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--test_reps", type=int, default=30)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.episodes, a.val_every, a.test_reps = 60, 30, 2
    from twin.paths import results_dir
    res = results_dir(ROOT)
    m5_path = os.environ.get("SCTWIN_M5", "")
    print(f"Tuning {len(CONFIGS)} Q-learning settings x {a.episodes} episodes, {a.parallel} in parallel"
          f" | demand: {'M5 real' if m5_path else 'synthetic'}", flush=True)
    jobs = [(n, c, a.episodes, a.val_every, a.test_reps, m5_path) for n, c in CONFIGS.items()]
    with mp.get_context("spawn").Pool(a.parallel) as pool:
        results = pool.map(run_config, jobs)
    summary = pd.DataFrame([r[0] for r in results]).sort_values("best_val_cost_L")
    pd.DataFrame([row for r in results for row in r[1]]).to_csv(
        os.path.join(res, "qlearning_tuning_curves.csv"), index=False)
    summary.to_csv(os.path.join(res, "qlearning_tuning_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summary[["config", "best_episode", "best_val_cost_L", "test_total_L",
                   "test_lost_pct", "test_recovery", "minutes"]].round(2).to_string(index=False))
    print("Setting chosen on VALIDATION cost:", summary.iloc[0].config)


if __name__ == "__main__":
    main()
