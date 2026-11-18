"""
Deep Q-Network (DQN) sourcing agent, written from scratch in NumPy (CPU).

Same job and same six actions as the tabular agent (twin/rl_agent.py), but the
state is a continuous 9-number vector and Q-values come from a neural network,
so the agent can generalise between situations it has never seen exactly.

Network      : 9 -> 64 -> 64 -> 6, ReLU, He initialisation
Learning     : Double DQN (van Hasselt et al., 2016) with experience replay,
               a target network synced every TARGET_SYNC steps, Huber loss,
               Adam optimiser
Exploration  : epsilon-greedy, epsilon set by the training script

A PyTorch twin of this agent for GPU training lives in
experiments/train_dqn_torch.py; it uses the same features(), actions and
hyper-parameters, so results are directly comparable.
"""
import numpy as np

from .rl_agent import execute_action, N_ACTIONS, ACTION_NAMES

N_FEATURES = 9
HP = dict(hidden=64, lr=5e-4, gamma=0.98, batch=64, buffer=100_000,
          train_every=2, target_sync=2000, warmup=2000, reward_scale=1e4)


def features(t, tw) -> np.ndarray:
    """Continuous state vector, each entry roughly in [0, 2]."""
    D = tw.mu["RM"]
    stores = sum(tw.on_hand[s] for s in tw.net.stores)
    dcs = sum(tw.on_hand[d] for d in tw.net.dcs)
    recent = tw.log["demand"][max(0, t - 7):t]
    return np.array([
        tw._ip("RM") / D / 30.0,
        tw.on_hand["RM"] / D / 30.0,
        (tw.on_hand["FG"] + tw.on_order["FG"]) / D / 10.0,
        dcs / D / 10.0,
        stores / D / 10.0,
        float(tw.dis.s1_disrupted(t)),
        sum(q for _, q in tw.backlog["S1"]) / D / 20.0,
        sum(q for _, q in tw.backlog["S2"]) / D / 20.0,
        (recent.mean() / D) if len(recent) else 1.0,
    ], dtype=np.float64)


class MLP:
    def __init__(self, sizes, rng):
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, X):
        acts = [X]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return acts

    def predict(self, X):
        return self.forward(X)[-1]

    def step(self, X, grad_out, lr):
        """Backpropagate dLoss/dOutput and take one Adam step."""
        acts = self.forward(X)
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        g = grad_out
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ g
            gb[i] = g.sum(0)
            if i > 0:
                g = (g @ self.W[i].T) * (acts[i] > 0)
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        params = self.W + self.b
        grads = gW + gb
        for k, (p, gr) in enumerate(zip(params, grads)):
            self.m[k] = b1 * self.m[k] + (1 - b1) * gr
            self.v[k] = b2 * self.v[k] + (1 - b2) * gr * gr
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + eps)

    def copy_from(self, other):
        self.W = [w.copy() for w in other.W]
        self.b = [b.copy() for b in other.b]


class DQNAgent:
    def __init__(self, seed=0, hp=None):
        self.hp = dict(HP, **(hp or {}))
        self.rng = np.random.default_rng(seed)
        sizes = [N_FEATURES, self.hp["hidden"], self.hp["hidden"], N_ACTIONS]
        self.net = MLP(sizes, self.rng)
        self.tgt = MLP(sizes, self.rng)
        self.tgt.copy_from(self.net)
        n = self.hp["buffer"]
        self.S = np.zeros((n, N_FEATURES)); self.A = np.zeros(n, int)
        self.R = np.zeros(n); self.S2 = np.zeros((n, N_FEATURES)); self.Dn = np.zeros(n)
        self.ptr = self.size = self.steps = 0
        self.epsilon = 1.0
        self.learning = True
        self.losses = []
        self.reset()

    # interface used by the simulation engine ---------------------------
    def reset(self):
        self.prev = None
        self.reward = 0.0
        self.action_counts = np.zeros(N_ACTIONS, int)

    def act(self, t, tw):
        s = features(t, tw)
        if self.learning and self.prev is not None:
            self._store(self.prev[0], self.prev[1], self.reward, s, 0.0)
            self._maybe_train()
        if self.learning and self.rng.random() < self.epsilon:
            a = int(self.rng.integers(N_ACTIONS))
        else:
            a = int(np.argmax(self.net.predict(s[None])[0]))
        self.prev = (s, a)
        self.action_counts[a] += 1
        return execute_action(a, tw)

    def feedback(self, r):
        self.reward = r

    def end_episode(self):
        if self.learning and self.prev is not None:
            self._store(self.prev[0], self.prev[1], self.reward, self.prev[0], 1.0)

    # learning ---------------------------------------------------------
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
        S, A, R, S2, Dn = self.S[idx], self.A[idx], self.R[idx], self.S2[idx], self.Dn[idx]
        a_star = np.argmax(self.net.predict(S2), 1)                       # Double DQN
        q_next = self.tgt.predict(S2)[np.arange(len(idx)), a_star]
        y = R + hp["gamma"] * (1 - Dn) * q_next
        q = self.net.predict(S)
        err = q[np.arange(len(idx)), A] - y
        g = np.zeros_like(q)
        g[np.arange(len(idx)), A] = np.clip(err, -1, 1) / len(idx)          # Huber gradient
        self.net.step(S, g, hp["lr"])
        self.losses.append(float(np.mean(np.minimum(err ** 2 / 2, np.abs(err) - 0.5))))
        if self.steps % hp["target_sync"] == 0:
            self.tgt.copy_from(self.net)

    # persistence ------------------------------------------------------
    def save(self, path):
        np.savez(path, *self.net.W, *self.net.b)

    def load(self, path):
        z = np.load(path)
        k = len(self.net.W)
        self.net.W = [z[f"arr_{i}"] for i in range(k)]
        self.net.b = [z[f"arr_{i}"] for i in range(k, 2 * k)]
        self.tgt.copy_from(self.net)
