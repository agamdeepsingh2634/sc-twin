"""
GPU version of the Double-DQN sourcing agent (PyTorch).

Same state features, actions, network size and hyper-parameters as the NumPy
agent in twin/dqn.py, so results are directly comparable; the only difference
is that the network trains on a GPU (CUDA or Apple MPS) when one is present,
which allows a bigger network and more training episodes.

Setup on the laptop (NVIDIA GPU):
    pip install numpy pandas scipy matplotlib
    pip install torch --index-url https://download.pytorch.org/whl/cu121
Check:  python -c "import torch; print(torch.cuda.is_available())"

Run (from the sc_twin folder):
    python -m experiments.prepare_m5 --sales path/to/sales_train_evaluation.csv --calendar path/to/calendar.csv
    SCTWIN_M5=results/m5_foods_daily.npz python -m experiments.train_dqn_torch --episodes 10000 --hidden 256
  (Windows PowerShell: $env:SCTWIN_M5="results/m5_foods_daily.npz"; python -m experiments.train_dqn_torch ...)

Outputs (suffix _m5 when real data is on): results/dqn_torch_training*.csv,
results/dqn_torch_test*.csv, results/dqn_torch_weights*.pt
"""
import argparse, os, sys, time
import numpy as np, pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from twin import SupplyChainTwin, SCENARIOS, sample_scenario, summarize, m5
from twin.disruptions import DisruptionState
from twin.optimize import make_cases
from twin.dqn import features, N_FEATURES, HP
from twin.rl_agent import execute_action, N_ACTIONS, ACTION_NAMES
from experiments.run_sensitivity import knee_policy

from twin.paths import results_dir
RES = results_dir(ROOT)
SCEN = list(SCENARIOS)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class QNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(N_FEATURES, hidden), nn.ReLU(),
                               nn.Linear(hidden, hidden), nn.ReLU(),
                               nn.Linear(hidden, N_ACTIONS))

    def forward(self, x):
        return self.f(x)


class TorchDQNAgent:
    """Drop-in replacement for twin.dqn.DQNAgent (same engine interface)."""

    def __init__(self, hidden, device, seed=0, hp=None):
        self.hp = dict(HP, **(hp or {}))
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.dev = device
        self.net = QNet(hidden).to(device)
        self.tgt = QNet(hidden).to(device)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.hp["lr"])
        n = self.hp["buffer"]
        self.S = np.zeros((n, N_FEATURES), np.float32); self.A = np.zeros(n, np.int64)
        self.R = np.zeros(n, np.float32); self.S2 = np.zeros((n, N_FEATURES), np.float32)
        self.Dn = np.zeros(n, np.float32)
        self.ptr = self.size = self.steps = 0
        self.epsilon, self.learning, self.losses = 1.0, True, []
        self.reset()

    def reset(self):
        self.prev, self.reward = None, 0.0
        self.action_counts = np.zeros(N_ACTIONS, int)

    def act(self, t, tw):
        s = features(t, tw).astype(np.float32)
        if self.learning and self.prev is not None:
            self._store(self.prev[0], self.prev[1], self.reward, s, 0.0)
            self._maybe_train()
        if self.learning and self.rng.random() < self.epsilon:
            a = int(self.rng.integers(N_ACTIONS))
        else:
            with torch.no_grad():
                a = int(self.net(torch.from_numpy(s).to(self.dev)[None]).argmax(1).item())
        self.prev = (s, a)
        self.action_counts[a] += 1
        return execute_action(a, tw)

    def feedback(self, r):
        self.reward = r

    def end_episode(self):
        if self.learning and self.prev is not None:
            self._store(self.prev[0], self.prev[1], self.reward, self.prev[0], 1.0)

    def _store(self, s, a, r, s2, done):
        i = self.ptr
        self.S[i], self.A[i], self.R[i], self.S2[i], self.Dn[i] = s, a, r / self.hp["reward_scale"], s2, done
        self.ptr = (self.ptr + 1) % self.hp["buffer"]
        self.size = min(self.size + 1, self.hp["buffer"])

    def _maybe_train(self):
        self.steps += 1
        hp = self.hp
        if self.size < hp["warmup"] or self.steps % hp["train_every"]:
            return
        idx = self.rng.integers(0, self.size, hp["batch"])
        T = lambda x: torch.from_numpy(x[idx]).to(self.dev)
        S, A, R, S2, Dn = T(self.S), T(self.A), T(self.R), T(self.S2), T(self.Dn)
        with torch.no_grad():
            a_star = self.net(S2).argmax(1, keepdim=True)                      # Double DQN
            y = R + hp["gamma"] * (1 - Dn) * self.tgt(S2).gather(1, a_star).squeeze(1)
        q = self.net(S).gather(1, A[:, None]).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.losses.append(float(loss.item()))
        if self.steps % hp["target_sync"] == 0:
            self.tgt.load_state_dict(self.net.state_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10000)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--test_reps", type=int, default=30)
    a = ap.parse_args()
    dev = pick_device()
    suffix = "_m5" if m5.enabled() else ""
    print(f"device: {dev} | demand: {'M5 real' if suffix else 'synthetic'} | episodes: {a.episodes}")
    kp, _ = knee_policy()
    agent = TorchDQNAgent(a.hidden, dev, hp=dict(batch=a.batch))
    rng = np.random.default_rng(1)
    hist, t0 = [], time.time()
    for ep in range(a.episodes):
        agent.epsilon = max(0.05, 1.0 - ep / (0.6 * a.episodes))
        agent.reset()
        s = int(rng.integers(1e9))
        ev = sample_scenario(SCEN[ep % len(SCEN)], np.random.default_rng(s))
        log = SupplyChainTwin(kp, ev, seed=s, rm_agent=agent).run()
        agent.end_episode()
        m = summarize(log, DisruptionState(ev).first_start)
        hist.append(dict(episode=ep + 1, cost_total_L=m["cost_total"] / 1e5, fill_rate=m["fill_rate"]))
        if (ep + 1) % max(1, a.episodes // 20) == 0:
            h = pd.DataFrame(hist[-200:])
            print(f"ep {ep+1:6d}  eps={agent.epsilon:.2f}  cost={h.cost_total_L.mean():.2f}L  "
                  f"fill={h.fill_rate.mean()*100:.2f}%  ({time.time()-t0:.0f}s)", flush=True)
    pd.DataFrame(hist).to_csv(os.path.join(RES, f"dqn_torch_training{suffix}.csv"), index=False)
    torch.save(agent.net.state_dict(), os.path.join(RES, f"dqn_torch_weights{suffix}.pt"))

    agent.learning = False
    rows = []
    for seed, ev in make_cases(a.test_reps, base_seed=900_000):
        st = DisruptionState(ev).first_start
        for name, ag in [("Optimised (knee)", None), ("Deep Q-network (GPU)", agent)]:
            if ag:
                ag.reset()
            m = summarize(SupplyChainTwin(kp, ev, seed=seed, rm_agent=ag).run(), st)
            m.update(policy=name, scenario=ev[0]["label"] if ev else "No disruption")
            rows.append(m)
    test = pd.DataFrame(rows)
    test.to_csv(os.path.join(RES, f"dqn_torch_test{suffix}.csv"), index=False)
    print(test.groupby("policy", sort=False).agg(
        total_L=("cost_total", lambda x: x.mean() / 1e5),
        lost=("fill_rate", lambda x: 100 * (1 - x.mean())),
        rec=("recovery_time", "mean")).round(2))
    print(f"done in {time.time() - t0:.0f}s on {dev}")


if __name__ == "__main__":
    main()
