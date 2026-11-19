# Supply-Chain Disruption Digital Twin, v1.0 (final)

**B.Tech Production Engineering final-year project, VJTI (Semester VII, R5PE4801P)**
Agamdeep Singh · Aman Raju · Vinay Sethia · Tisha Rathi
Guide: Dr. D. N. Raut

A simulation-based digital twin of an FMCG retail supply chain. It stress-tests the chain against disruptions, finds cost-vs-resilience-optimal inventory policies with a multi-objective genetic algorithm (NSGA-II), and benchmarks two reinforcement-learning agents. Everything is evaluated on synthetic demand and on real Walmart sales data (Kaggle M5).

## Headline results (210 held-out test runs per method)

| Method | Synthetic demand | Real Walmart M5 demand |
|---|---|---|
| Lean (JIT) | ₹24.02 L, 5.19% lost | ₹18.27 L, 2.68% lost |
| Lean + dual sourcing | ₹24.20 L, 3.81% lost | ₹18.67 L, 1.37% lost |
| Buffered | ₹19.32 L, 0.76% lost | ₹18.70 L, 0.24% lost |
| **Optimised policy (NSGA-II knee)** | **₹16.77 L**, 1.94% lost | **₹14.69 L**, 1.61% lost |
| Q-learning, tuned | ₹18.94 L, 0.95% lost | ₹15.89 L, 0.96% lost |
| Deep Q-network, tuned (GPU) | ₹17.04 L, 1.56% lost | ₹15.17 L, 1.58% lost |

Total cost covers about 10 months and includes ₹250 per lost case. See `results/final_comparison.csv` and `results/fig14_final_comparison.png`.

- **Optimised policy:** same service as Lean for 26% (synthetic) or 18% (real) less operating cost. The synthetic-optimised policy still beats every standard strategy by about 13% on real data (transfer test).
- **Tuning matters:** the deep Q-network went from ₹29.06 L untuned (last snapshot) to ₹17.04 L tuned with best-checkpoint selection (−41%). Tabular Q-learning improved by 6% (synthetic) and 11% (real).
- **Sensitivity:**
  - The optimised policy stays 5–22% cheaper than the best reference under every cost assumption tested.
  - It must be re-optimised if the import lead time reaches 15 days or more.
  - A backup supplier pays only below about ₹31–41 per case (60% capacity) or ₹60–71 (80–100%).
- **ANOVA:** the inventory strategy explains 88–94% of cost variation; the disruption type explains only 2–5%.
- **Reproducibility:** all experiments were re-run on a second computer (Windows, RTX 3050) and reproduced the original results exactly.

## Model
```
S1 primary supplier (imported via port, 10 d, cheap) ┐
                                                     ├─> Factory (RM → FG) ─┬─> DC North ─> 3 stores
S2 backup supplier (domestic, 4 d, +₹120, 60% cap)   ┘                     └─> DC West  ─> 3 stores
```
- Daily time-step simulation, written from scratch (`twin/engine.py`).
- (s, S) policies in days of cover. Lost sales at stores, backorders upstream.
- **7 disruption scenarios** (`twin/disruptions.py`): supplier outage, port blockage, demand surge, factory capacity loss, DC flood, compound, none.
- **Metrics** (`twin/metrics.py`): fill rate, cost breakdown, time-to-impact, recovery time, resilience-triangle loss, bullwhip.
- **Demand:**
  - Synthetic M5-style generator (`twin/demand.py`).
  - Real Kaggle M5 FOODS sales (`twin/m5.py`): CA_1–3 feed DC North, TX_1–3 feed DC West, with each one-year window rescaled.

## Methods
| Method | Code | Notes |
|---|---|---|
| Monte Carlo baseline | `experiments/run_baseline.py` | 7 scenarios × 4 policies × 200 runs, common random numbers, Wilcoxon paired tests |
| NSGA-II simulation-optimisation | `twin/optimize.py`, `experiments/run_optimization.py` | Written from scratch; 9 variables; lost sales ≤ 10%; archive; tested on 210 fresh runs |
| Sensitivity, ANOVA, node criticality | `run_sensitivity.py`, `run_extras.py` | |
| Transfer test | `run_transfer.py` | Synthetic-optimised policy evaluated on real demand |
| Tabular Q-learning | `twin/rl_agent.py`, `tune_qlearning.py` | 6 settings × 14,000 episodes; best validation checkpoint |
| Deep Q-network (Double DQN) | `twin/dqn.py` (NumPy), `train_dqn_torch.py`, `tune_dqn_torch.py` (PyTorch, GPU) | 4 settings × 6,000 episodes on an RTX 3050; best validation checkpoint |

**Evaluation protocol:**
- Training scenarios (seed base 1000), validation scenarios (700000, 35 runs) and test scenarios (900000, 210 runs) never overlap.
- All reported numbers are test numbers.

## Running it
**Windows, one click:**
- `QUICK_TEST.bat`: a 4-minute smoke test
- `RUN_ON_MY_PC.bat`: the full study (about 3 hours, GPU used for deep RL)
- `TUNE_QLEARNING.bat` (about 40 minutes) and `TUNE_DQN.bat` (about 5 hours)
- `KEEP_AWAKE.bat`: stops the PC from sleeping during long runs

**Manual run** (Python 3.11–3.13):
```
pip install -r requirements.txt          # plus torch for the GPU scripts
python run_all.py                         # everything; add --quick for a test
set SCTWIN_M5=results\m5_foods_daily.npz  # Windows: switch to real demand
```
The Walmart M5 data are not included in this repository (Kaggle's competition terms). Download `sales_train_evaluation.csv` and `calendar.csv` from the [M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) competition, then build the compact file `results/m5_foods_daily.npz` with:
```
python -m experiments.prepare_m5 --sales sales_train_evaluation.csv --calendar calendar.csv
```
All real-demand results are already included in `results/m5/`, so this is only needed to re-run them.

## Folder guide
- `results/`: synthetic-demand results. `results/m5/`: real-demand results.
- `fig1`–`fig13`: baseline, frontier, sensitivity, criticality and demand figures. `fig14`: final comparison. `fig15`: DQN tuning curves.
- `*_tuning_summary.csv`, `*_tuning_curves.csv`, `dqn_tuned_*.pt`, `qlearning_tuned_*.pkl`: tuned RL agents.
- `dashboard/index.html`: interactive dashboard (opens offline).
- `logs/`: console logs of the full laptop runs.

## Limitations and future work
- One aggregate product family; multi-SKU is future work.
- Costs and lead times are literature-based assumptions; sensitivity analysis covers them.
- The RL agents control raw-material sourcing only; the other echelons follow the optimised policy.
- The simulator runs on the CPU, so RL training is environment-bound. A GPU-vectorised simulator would allow much larger deep-RL studies.
- NSGA-II hypervolume was still rising slightly at generation 60.

## AI assistance
The code, analysis and documentation were developed with the help of Claude (Anthropic), with the knowledge and approval of the project guide. This is declared in the project report.

## Licence
Released under the [MIT Licence](LICENSE). The Walmart M5 data belong to their owners and are not redistributed here.


## Real-data replay (future scope)

`realdata/` replays 33 months of real DataCo orders with real lead times from Olist and USAID SCMS through the twin's reorder logic. See `realdata/README.md`.
