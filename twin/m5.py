"""
Real demand from the Kaggle M5 Forecasting dataset (Walmart, 2011-2016).

Pipeline
  1. prepare(): read sales_train_evaluation.csv (+ calendar.csv), keep the
     FOODS category (FMCG), sum units per store per day, and save a compact
     file results/m5_foods_daily.npz (10 stores x 1941 days). This is done once.
  2. Store mapping: our six stores take the shape of six real Walmart stores
       N1, N2, N3 <- CA_1, CA_2, CA_3   (served by DC North)
       W1, W2, W3 <- TX_1, TX_2, TX_3   (served by DC West)
     Each 365-day window is rescaled so its mean equals our store's base mean
     (cost parameters stay comparable, multi-year growth is removed), while
     the weekly pattern, SNAP-day
     spikes, holidays, promotions, trends and real noise are kept exactly.
  3. Windows: a simulation year is a 365-day window of the real series. Run
     with seed s uses window (s mod W), with W windows spaced 30 days apart,
     so replications cover different real years and seasons.

Switch the whole model to M5 demand with
     from twin import m5; m5.enable("results/m5_foods_daily.npz")
or set the environment variable SCTWIN_M5=<path to npz> before running any
experiment script.
"""
import os
import numpy as np

STORE_MAP = {"N1": "CA_1", "N2": "CA_2", "N3": "CA_3",
             "W1": "TX_1", "W2": "TX_2", "W3": "TX_3"}
CATEGORY = "FOODS"
WINDOW_STEP = 30

_STATE = {"data": None}


def prepare(sales_csv: str, calendar_csv: str = None, out_npz: str = "results/m5_foods_daily.npz",
            category: str = CATEGORY) -> str:
    import pandas as pd
    head = pd.read_csv(sales_csv, nrows=0).columns
    dcols = [c for c in head if c.startswith("d_")]
    usecols = ["store_id", "cat_id"] + dcols
    dtypes = {c: "int32" for c in dcols}
    df = pd.read_csv(sales_csv, usecols=usecols, dtype=dtypes)
    df = df[df.cat_id == category]
    agg = df.groupby("store_id")[dcols].sum()
    dates = None
    if calendar_csv and os.path.exists(calendar_csv):
        cal = pd.read_csv(calendar_csv, usecols=["d", "date"]).set_index("d")
        dates = np.array([str(x) for x in cal.loc[dcols, "date"]], dtype="U10")
    os.makedirs(os.path.dirname(out_npz) or ".", exist_ok=True)
    np.savez_compressed(out_npz, stores=np.array([str(x) for x in agg.index], dtype="U8"),
                        sales=agg.values.astype(np.float32),
                        dates=dates if dates is not None else np.array([], dtype="U10"))
    return out_npz


def enable(npz_path: str):
    z = np.load(npz_path, allow_pickle=False)
    stores = list(z["stores"])
    sales = z["sales"]
    _STATE["data"] = {s: sales[stores.index(s)] for s in stores}
    _STATE["dates"] = z["dates"]
    _STATE["path"] = npz_path


def disable():
    _STATE["data"] = None


def enabled() -> bool:
    if _STATE["data"] is None and os.environ.get("SCTWIN_M5"):
        enable(os.environ["SCTWIN_M5"])
    return _STATE["data"] is not None


def n_windows(horizon: int = 365) -> int:
    n = len(next(iter(_STATE["data"].values())))
    return (n - horizon) // WINDOW_STEP + 1


def window_start(seed: int, horizon: int = 365) -> int:
    return (seed % n_windows(horizon)) * WINDOW_STEP


def demand_for(seed: int, store_means: dict, horizon: int = 365) -> dict:
    """Real demand window for this seed, rescaled to our store means."""
    start = window_start(seed, horizon)
    out = {}
    for ours, real in STORE_MAP.items():
        win = _STATE["data"][real][start:start + horizon]
        scale = store_means[ours] / win.mean()     # remove multi-year level drift
        out[ours] = np.round(win * scale).astype(float)
    return out


def window_dates(seed: int, horizon: int = 365):
    d = _STATE.get("dates")
    if d is None or len(d) == 0:
        return None
    s = window_start(seed, horizon)
    return d[s], d[s + horizon - 1]
