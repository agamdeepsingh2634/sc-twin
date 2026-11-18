"""
Deep Q-network tuning on the GPU, with best-checkpoint selection.

Why: a DQN's performance swings during training, so the weights at the LAST
episode are often not the best ones. This script fixes that and tunes the
main settings:

  * every VAL_EVERY episodes the agent is frozen and scored on a fixed
    VALIDATION set (35 runs, seeds never used for testing); the best weights
    are kept (early-stopping / checkpoint selection)
  * several configurations train in parallel (one process each, sharing the GPU)
  * the best checkpoint of each configuration is finally scored on the same
    210 held-out TEST runs used for every other method

Usage (from the sc_twin folder):
    python -m experiments.tune_dqn_torch                 # synthetic demand
    set SCTWIN_M5=results\\m5_foods_daily.npz  (Windows) then run again for M5
    python -m experiments.tune_dqn_torch --quick         # 2-minute smoke test

Outputs (in results/ or results/m5/):
    dqn_tuning_summary.csv   one row per configuration (validation + test)
    dqn_tuning_curves.csv    validation cost every VAL_EVERY episodes
    dqn_tuned_<cfg>.pt       best weights per configuration
"""
import argparse, copy, os, sys, time
import multiprocessing as mp
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CONFIGS = {
    # name: (hidden, lr, batch, target_sync, gamma, explore_frac)
    "A_small_64":     (64,  5e-4,  64, 2000, 0.98, 0.6),
    "B_medium_128":   (128, 2.5e-4, 128, 3000, 0.98, 0.6),
    "C_large_256_slow": (256, 1e-4, 256, 4000, 0.98, 0.6),
    "D_128_longexplore": (128, 2.5e-4, 128, 3000, 0.99, 0.8),
}


def run_config(args):
    name, cfg, episodes, val_every, test_reps, m5_path, quick = args
    if m5_path:
        os.environ["SCTWIN_M5"] = m5_path
    import torch
    from twin import SupplyChainTwin, SCENARIOS, sample_scenario, summarize
    from twin.disruptions import DisruptionState
    from twin.optimize import make_cases
    from twin.paths import results_dir
    from experiments.run_sensitivity import knee_policy
    from experiments.train_dqn_torch import TorchDQNAgent, pick_device

    hidden, lr, batch, sync, gamma, explore = cfg
    dev = pick_device()
    res = results_dir(ROOT)
    kp, _ = knee_policy()
    scen = list(SCENARIOS)
    agent = TorchDQNAgent(hidden, dev, seed=0,
                          hp=dict(lr=lr, batch=batch, target_sync=sync, gamma=gamma))
    val_cases = make_cases(1 if quick else 5, base_seed=700_000)     # never used for testing

    def score(cases):
        agent.learning = False
        rows = []
        for seed, ev in cases:
            agent.reset()
            m = summarize(SupplyChainTwin(kp, ev, seed=seed, rm_agent=agent).run(),
                          DisruptionState(ev).first_start)
            rows.append(m)
        agent.learning = True
        return rows

    rng = np.random.default_rng(1)
    best_val, best_state, best_ep, curve = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(episodes):
        agent.epsilon = max(0.05, 1.0 - ep / (explore * episodes))
        agent.reset()
        s = int(rng.integers(1e9))
        ev = sample_scenario(scen[ep % len(scen)], np.random.default_rng(s))
        SupplyChainTwin(kp, ev, seed=s, rm_agent=agent).run()
        agent.end_episode()
        if (ep + 1) % val_every == 0:
            rows = score(val_cases)
            v = float(np.mean([r["cost_total"] for r in rows])) / 1e5
            lost = 100 * (1 - float(np.mean([r["fill_rate"] for r in rows])))
            curve.append(dict(config=name, episode=ep + 1, val_cost_L=v, val_lost_pct=lost,
                              epsilon=agent.epsilon))
            if v < best_val:
                best_val, best_ep = v, ep + 1
                best_state = copy.deepcopy(agent.net.state_dict())
            print(f"  [{name}] ep {ep+1:5d} eps={agent.epsilon:.2f} val={v:6.2f}L lost={lost:5.2f}% "
                  f"best={best_val:6.2f}L@{best_ep} ({time.time()-t0:.0f}s)", flush=True)

    agent.net.load_state_dict(best_state)
    torch.save(best_state, os.path.join(res, f"dqn_tuned_{name}.pt"))
    test = score(make_cases(test_reps, base_seed=900_000))
    out = dict(config=name, hidden=hidden, lr=lr, batch=batch, target_sync=sync, gamma=gamma,
               explore_frac=explore, episodes=episodes, best_episode=best_ep, best_val_cost_L=best_val,
               test_total_L=np.mean([r["cost_total"] for r in test]) / 1e5,
               test_operating_L=np.mean([r["cost_operating"] for r in test]) / 1e5,
               test_lost_pct=100 * (1 - np.mean([r["fill_rate"] for r in test])),
               test_recovery=np.nanmean([r["recovery_time"] for r in test]),
               minutes=(time.time() - t0) / 60, device=str(dev))
    return out, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=6000)
    ap.add_argument("--val_every", type=int, default=250)
    ap.add_argument("--test_reps", type=int, default=30)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.episodes, a.val_every, a.test_reps = 40, 20, 2
    from twin.paths import results_dir
    res = results_dir(ROOT)
    m5_path = os.environ.get("SCTWIN_M5", "")
    print(f"Tuning {len(CONFIGS)} DQN configurations x {a.episodes} episodes, "
          f"{a.parallel} in parallel | demand: {'M5 real' if m5_path else 'synthetic'}", flush=True)
    jobs = [(n, c, a.episodes, a.val_every, a.test_reps, m5_path, a.quick) for n, c in CONFIGS.items()]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(a.parallel, len(jobs))) as pool:
        results = pool.map(run_config, jobs)
    summary = pd.DataFrame([r[0] for r in results]).sort_values("best_val_cost_L")
    curves = pd.DataFrame([row for r in results for row in r[1]])
    summary.to_csv(os.path.join(res, "dqn_tuning_summary.csv"), index=False)
    curves.to_csv(os.path.join(res, "dqn_tuning_curves.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summary[["config", "best_episode", "best_val_cost_L", "test_total_L",
                   "test_lost_pct", "test_recovery", "minutes"]].round(2).to_string(index=False))
    print("Configuration chosen on VALIDATION cost:", summary.iloc[0].config)


if __name__ == "__main__":
    main()
