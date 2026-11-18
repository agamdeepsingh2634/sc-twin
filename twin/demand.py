"""
Synthetic daily demand generator, shaped like Walmart M5 FMCG data:
  * day-of-week pattern (weekend peak)
  * mild annual seasonality + Diwali festive peak (late Oct / early Nov)
  * random short promotions (+40% for 3-7 days)
  * over-dispersed counts (negative binomial noise)

Later this can be swapped for real M5 / Kaggle series via `load_m5_series`.
"""
import numpy as np

WEEKLY = np.array([0.90, 0.85, 0.90, 0.95, 1.05, 1.20, 1.15])  # Mon..Sun


def seasonal_profile(n_days: int, start_doy: int = 0, start_dow: int = 0) -> np.ndarray:
    d = np.arange(n_days)
    dow = (d + start_dow) % 7
    doy = (d + start_doy) % 365
    weekly = WEEKLY[dow]
    annual = 1.0 + 0.08 * np.sin(2 * np.pi * (doy - 80) / 365)
    festive = 1.0 + 0.35 * np.exp(-0.5 * ((doy - 305) / 10.0) ** 2)  # Diwali
    return weekly * annual * festive


def generate_store_demand(store_means: dict, n_days: int, rng: np.random.Generator,
                          promo_per_month: float = 1.0, dispersion: float = 25.0,
                          start_doy: int = 0) -> dict:
    """Return {store: np.array(n_days)} of integer daily demand."""
    base = seasonal_profile(n_days, start_doy=start_doy)
    out = {}
    for store, mean in store_means.items():
        promo = np.ones(n_days)
        n_promos = rng.poisson(promo_per_month * n_days / 30.0)
        for start in rng.integers(0, n_days, n_promos):
            promo[start:start + rng.integers(3, 8)] *= 1.4
        mu = mean * base * promo
        p = dispersion / (dispersion + mu)
        out[store] = rng.negative_binomial(dispersion, p).astype(float)
    return out


def load_m5_series(path: str, item_ids: list, store_map: dict):
    """Placeholder for phase 2: read M5 'sales_train_evaluation.csv' and map
    selected FOODS/HOUSEHOLD items to our six stores. Implemented once the
    dataset is uploaded."""
    raise NotImplementedError("Upload the M5 dataset to enable real demand.")
