"""
Run NSGA-II over the (s, S) parameters of every echelon.

Training : 7 scenarios x R_TRAIN runs per candidate (fixed seeds, CRN)
Testing  : every Pareto solution + the 3 reference policies are re-evaluated
           on FRESH seeds (7 x R_TEST) -> guards against over-fitting to the
           training scenarios ("optimiser's curse").

Outputs: results/opt_history.csv, opt_final_population.csv,
         opt_pareto_test.csv, opt_reference_test.csv

Usage:  python -m experiments.run_optimization --pop 40 --gen 50
"""
import argparse, os, sys, time
from multiprocessing import Pool
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from twin import REFERENCE_POLICIES
from twin.optimize import (BOUNDS, VAR_NAMES, decode, make_cases, evaluate_policy,
                           objectives, nsga2, hypervolume_2d, fast_nondominated_sort)

from twin.paths import results_dir
RES = results_dir(ROOT)
TRAIN = None
HV_REF = np.array([25.0, 12.0])   # (cost lakh, lost-sales %) reference point


def _init(reps):
    global TRAIN
    TRAIN = make_cases(reps, base_seed=1000)


def f_train(x):
    return objectives(evaluate_policy(decode(x), TRAIN))


def encode(p):
    v = np.array([p.store[0], p.store[1] - p.store[0], p.dc[0], p.dc[1] - p.dc[0],
                  p.fg[0], p.fg[1] - p.fg[0], p.rm[0], p.rm[1] - p.rm[0],
                  1.0 if p.dual_sourcing else 0.0])
    return (v - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])


def test_eval(args):
    x, name, reps = args
    p = decode(x, name)
    m = evaluate_policy(p, make_cases(reps, base_seed=900_000))   # fresh seeds
    m.update(p.as_dict())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--gen", type=int, default=50)
    ap.add_argument("--train_reps", type=int, default=3)
    ap.add_argument("--test_reps", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    hist = []

    def cb(g, X, F):
        front = F[fast_nondominated_sort(F)[0]]
        hv = hypervolume_2d(front, HV_REF)
        hist.append(dict(gen=g + 1, hypervolume=hv, n_front=len(front),
                         min_cost=F[:, 0].min(), min_lost=F[:, 1].min(),
                         elapsed_s=time.time() - t0))
        print(f"gen {g + 1:3d}  HV={hv:7.2f}  front={len(front):2d}  "
              f"minCost={F[:, 0].min():.2f}L  minLost={F[:, 1].min():.3f}%  "
              f"({time.time() - t0:.0f}s)", flush=True)

    seeds_x = np.array([encode(p) for p in REFERENCE_POLICIES])
    with Pool(os.cpu_count(), initializer=_init, initargs=(a.train_reps,)) as pool:
        PX, PF, X, F = nsga2(f_train, len(BOUNDS), a.pop, a.gen, seed=7,
                             seeds_x=seeds_x, callback=cb, map_fn=pool.map)

        pd.DataFrame(hist).to_csv(os.path.join(RES, "opt_history.csv"), index=False)
        fp = pd.DataFrame(X, columns=VAR_NAMES)
        fp["train_cost_L"], fp["train_lost_pct"] = F[:, 0], F[:, 1]
        fp.to_csv(os.path.join(RES, "opt_final_population.csv"), index=False)

        # de-duplicate Pareto set, then test out-of-sample
        _, idx = np.unique(np.round(PF, 4), axis=0, return_index=True)
        PX, PF = PX[idx], PF[idx]
        jobs = [(x, f"P{i:02d}", a.test_reps) for i, x in enumerate(PX)]
        jobs += [(encode(p), p.name, a.test_reps) for p in REFERENCE_POLICIES]
        out = pool.map(test_eval, jobs)

    test = pd.DataFrame(out)
    test["train_cost_L"] = list(PF[:, 0]) + [np.nan] * len(REFERENCE_POLICIES)
    test["train_lost_pct"] = list(PF[:, 1]) + [np.nan] * len(REFERENCE_POLICIES)
    test["test_cost_L"] = test.cost_operating / 1e5
    test["test_lost_pct"] = 100 * (1 - test.fill_rate)
    n = len(PX)
    test.iloc[:n].to_csv(os.path.join(RES, "opt_pareto_test.csv"), index=False)
    test.iloc[n:].to_csv(os.path.join(RES, "opt_reference_test.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s; {n} Pareto solutions")


if __name__ == "__main__":
    main()
