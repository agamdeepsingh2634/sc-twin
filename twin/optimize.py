"""
Simulation-optimisation: NSGA-II (Deb et al., 2002), written from scratch.

Decision variables (9), all in days of cover:
    store_s, store_gap, dc_s, dc_gap, fg_s, fg_gap, rm_s, rm_gap, dual
  where S = s + gap (this encoding guarantees S > s), and dual = 1 switches
  raw-material orders to the backup supplier when S1 is disrupted.

Objectives (both minimised), averaged over a fixed "stress-test mix" of
disruption scenarios with common random numbers:
    f1 = expected OPERATING cost  (holding + ordering + backup premium), INR lakh
         -> what you PAY for protection
    f2 = expected LOST-SALES %    (100 * (1 - fill rate))
         -> how much service you LOSE when things go wrong
Constraint: f2 <= MAX_LOST_PCT (10%), handled with Deb's constrained-domination
rule, so the search concentrates on service levels a real business would accept.
The trade-off curve between f1 and f2 is the Pareto frontier:
"how much resilience does each extra rupee buy?"
"""
import numpy as np

from .config import Policy
from .engine import SupplyChainTwin
from .disruptions import SCENARIOS, sample_scenario, DisruptionState
from .metrics import summarize

# (lower, upper) bounds in days of cover
BOUNDS = np.array([
    [1.0, 8.0],    # store s
    [1.0, 8.0],    # store gap (S - s)
    [1.0, 15.0],   # dc s
    [1.0, 12.0],   # dc gap
    [1.0, 15.0],   # fg s
    [1.0, 12.0],   # fg gap
    [5.0, 45.0],   # rm s
    [3.0, 25.0],   # rm gap
    [0.0, 1.0],    # dual sourcing flag (>= 0.5 -> on)
])
VAR_NAMES = ["store_s", "store_gap", "dc_s", "dc_gap", "fg_s", "fg_gap",
             "rm_s", "rm_gap", "dual"]


def decode(x, name="candidate") -> Policy:
    v = BOUNDS[:, 0] + x * (BOUNDS[:, 1] - BOUNDS[:, 0])
    return Policy(name,
                  store=(v[0], v[0] + v[1]), dc=(v[2], v[2] + v[3]),
                  fg=(v[4], v[4] + v[5]), rm=(v[6], v[6] + v[7]),
                  dual_sourcing=bool(v[8] >= 0.5))


def make_cases(reps: int, base_seed: int, severity: float = 1.0):
    """Fixed list of (seed, events) covering every scenario `reps` times."""
    cases = []
    for si, scen in enumerate(SCENARIOS):
        for r in range(reps):
            seed = base_seed + 10_000 * si + r
            cases.append((seed, sample_scenario(scen, np.random.default_rng(seed + 7),
                                                severity=severity)))
    return cases


def evaluate_policy(policy: Policy, cases, net=None, cost=None) -> dict:
    ms = []
    for seed, events in cases:
        log = SupplyChainTwin(policy, events, seed=seed, net=net, cost=cost).run()
        ms.append(summarize(log, DisruptionState(events).first_start))
    keys = ms[0].keys()
    return {k: float(np.nanmean([m[k] for m in ms])) for k in keys}


def objectives(m: dict):
    return np.array([m["cost_operating"] / 1e5, 100.0 * (1.0 - m["fill_rate"])])


# ---------------------------------------------------------------- NSGA-II core
MAX_LOST_PCT = 10.0   # service constraint: solutions losing >10% of sales are infeasible


def violation(F):
    return np.maximum(0.0, F[:, 1] - MAX_LOST_PCT)


def _dominates(fp, fq, vp, vq):
    """Deb's constrained domination: feasible beats infeasible; among
    infeasible, smaller violation wins; among feasible, Pareto domination."""
    if vp == 0 and vq > 0:
        return True
    if vp > 0 and vq > 0:
        return vp < vq
    if vp > 0 and vq == 0:
        return False
    return bool(np.all(fp <= fq) and np.any(fp < fq))


def fast_nondominated_sort(F):
    n = len(F)
    V = violation(F)
    S = [[] for _ in range(n)]
    dom_count = np.zeros(n, int)
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if _dominates(F[p], F[q], V[p], V[q]):
                S[p].append(q)
            elif _dominates(F[q], F[p], V[q], V[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding_distance(F):
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(m):
        o = np.argsort(F[:, j])
        d[o[0]] = d[o[-1]] = np.inf
        span = F[o[-1], j] - F[o[0], j] or 1.0
        d[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / span
    return d


def rank_and_crowd(F):
    rank = np.zeros(len(F), int)
    crowd = np.zeros(len(F))
    for i, fr in enumerate(fast_nondominated_sort(F)):
        rank[fr] = i
        crowd[fr] = crowding_distance(F[fr])
    return rank, crowd


def tournament(rank, crowd, rng):
    a, b = rng.integers(0, len(rank), 2)
    if rank[a] != rank[b]:
        return a if rank[a] < rank[b] else b
    return a if crowd[a] > crowd[b] else b


def sbx(p1, p2, rng, eta=15, pc=0.9):
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > pc:
        return c1, c2
    for i in range(len(p1)):
        if rng.random() < 0.5 and abs(p1[i] - p2[i]) > 1e-12:
            u = rng.random()
            beta = (2 * u) ** (1 / (eta + 1)) if u <= 0.5 else (1 / (2 * (1 - u))) ** (1 / (eta + 1))
            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return np.clip(c1, 0, 1), np.clip(c2, 0, 1)


def poly_mutation(x, rng, eta=20, pm=None):
    pm = pm or 1.0 / len(x)
    y = x.copy()
    for i in range(len(x)):
        if rng.random() < pm:
            u = rng.random()
            d = (2 * u) ** (1 / (eta + 1)) - 1 if u < 0.5 else 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            y[i] = y[i] + d
    return np.clip(y, 0, 1)


def hypervolume_2d(F, ref):
    """Area dominated by front F (minimisation) up to reference point."""
    P = F[np.all(F < ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[np.argsort(P[:, 0])]
    hv, best_y = 0.0, ref[1]
    for x, y in P:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return hv


def nsga2(eval_fn, n_var, pop_size, n_gen, seed=0, seeds_x=None, callback=None,
          map_fn=map):
    rng = np.random.default_rng(seed)
    X = rng.random((pop_size, n_var))
    if seeds_x is not None:                    # inject known policies
        X[:len(seeds_x)] = seeds_x
    F = np.array(list(map_fn(eval_fn, X)))
    AX, AF = [X.copy()], [F.copy()]            # archive of everything evaluated
    for g in range(n_gen):
        rank, crowd = rank_and_crowd(F)
        kids = []
        while len(kids) < pop_size:
            p1 = X[tournament(rank, crowd, rng)]
            p2 = X[tournament(rank, crowd, rng)]
            c1, c2 = sbx(p1, p2, rng)
            kids += [poly_mutation(c1, rng), poly_mutation(c2, rng)]
        Xk = np.array(kids[:pop_size])
        Fk = np.array(list(map_fn(eval_fn, Xk)))
        AX.append(Xk); AF.append(Fk)
        X2, F2 = np.vstack([X, Xk]), np.vstack([F, Fk])
        rank, crowd = rank_and_crowd(F2)
        order = np.lexsort((-crowd, rank))[:pop_size]   # elitist survival
        X, F = X2[order], F2[order]
        if callback:
            callback(g, X, F)
    # final answer = non-dominated set of the whole archive (an elitist
    # population can drop good points through crowding; the archive cannot)
    AX, AF = np.vstack(AX), np.vstack(AF)
    feas = violation(AF) == 0
    AX, AF = AX[feas], AF[feas]
    front = fast_nondominated_sort(AF)[0]
    return AX[front], AF[front], X, F
