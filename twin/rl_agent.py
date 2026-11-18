"""
Tabular Q-learning agent for raw-material sourcing (written from scratch).

The agent replaces the (s, S) rule at the raw-material node only; stores, DCs
and the factory keep the optimised (knee) policy. Every day it observes a small
discretised state and picks one of six sourcing actions. Reward = minus the
day's total cost (holding + ordering + lost sales + backup premium).

State (6 x 4 x 2 x 3 = 144 states)
  rm_ip   : raw-material inventory position, days of cover
  fg      : finished-goods stock + work-in-process, days of cover
  s1_bad  : is the primary supplier visibly disrupted? (0/1)
  stuck   : unshipped orders waiting at S1, days of cover
Actions
  0 do nothing
  1 order from S1 up to 20 days     2 order from S1 up to 32 days
  3 order from S2 up to 15 days     4 order from S2 up to 25 days
  5 re-route S1's open orders to S2, then top up from S2 to 20 days
  (orders smaller than MIN_BATCH_DAYS of demand are skipped, so the agent
   cannot rack up a fixed ordering cost every single day)
Update rule (Watkins Q-learning):
  Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]
"""
import numpy as np

RM_BINS = [6, 10, 14, 20, 28]
FG_BINS = [2, 4, 7]
STUCK_BINS = [5, 15]
N_ACTIONS = 6
MIN_BATCH_DAYS = 8   # an order must cover at least 8 days of demand (economic batch)
TARGETS = {1: 20, 2: 32, 3: 15, 4: 25, 5: 20}   # order-up-to, days of cover
ACTION_NAMES = ["Wait", "S1 to 20d", "S1 to 32d", "S2 to 15d", "S2 to 25d", "Re-route to S2"]


class QSourcingAgent:
    def __init__(self, alpha=0.1, gamma=0.97, epsilon=0.2, seed=0, scale=1e4):
        self.Q = np.zeros((len(RM_BINS) + 1, len(FG_BINS) + 1, 2, len(STUCK_BINS) + 1, N_ACTIONS))
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.rng = np.random.default_rng(seed)
        self.scale = scale
        self.learning = True
        self.reset()

    def reset(self):
        self.prev = None
        self.reward = 0.0
        self.action_counts = np.zeros(N_ACTIONS, int)

    def observe(self, t, tw):
        mu = tw.mu["RM"]
        rm = tw._ip("RM") / mu
        fg = (tw.on_hand["FG"] + tw.on_order["FG"]) / mu
        stuck = sum(q for _, q in tw.backlog["S1"]) / mu
        return (int(np.digitize(rm, RM_BINS)), int(np.digitize(fg, FG_BINS)),
                int(tw.dis.s1_disrupted(t)), int(np.digitize(stuck, STUCK_BINS)))

    def act(self, t, tw):
        s = self.observe(t, tw)
        if self.learning and self.prev is not None:
            ps, pa = self.prev
            target = self.reward / self.scale + self.gamma * self.Q[s].max()
            self.Q[ps + (pa,)] += self.alpha * (target - self.Q[ps + (pa,)])
        if self.learning and self.rng.random() < self.epsilon:
            a = int(self.rng.integers(N_ACTIONS))
        else:
            a = int(np.argmax(self.Q[s]))
        self.prev = (s, a)
        self.action_counts[a] += 1
        return self._execute(a, tw)

    def feedback(self, r):
        self.reward = r

    def _execute(self, a, tw):
        return execute_action(a, tw)


def execute_action(a, tw):
    """Shared by the tabular and deep agents: turn an action index into
    (quantity ordered from S1, quantity ordered from S2)."""
    mu, ip = tw.mu["RM"], tw._ip("RM")
    if a == 0:
        return 0.0, 0.0
    if a == 5:
        if tw.backlog["S1"]:
            tw.backlog["S2"].extend(tw.backlog["S1"])
            tw.backlog["S1"].clear()
        return 0.0, max(0.0, TARGETS[5] * mu - ip)
    q = max(0.0, TARGETS[a] * mu - ip)
    if q < MIN_BATCH_DAYS * mu:          # batch rule: tiny orders are not placed
        return 0.0, 0.0
    return (q, 0.0) if a in (1, 2) else (0.0, q)
