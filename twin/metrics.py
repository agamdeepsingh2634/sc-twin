"""
Performance and resilience metrics computed from one simulation log.

Service
  fill_rate            : sales / demand over the measured period
Cost (INR)
  cost_total, cost_holding, cost_ordering, cost_lost_sales, cost_backup_premium
Resilience (after the first disruption starts; 7-day rolling fill rate, 95% target)
  time_to_impact       : days until rolling fill rate first drops below target
                         (empirical "time-to-survive"; NaN if it never drops)
  recovery_time        : days from the drop until it is back above target and
                         stays there for 14 days (0 if never dropped)
  impact_depth         : lowest rolling fill rate reached (1.0 = no impact)
  lost_sales_disr      : cases lost in [start, start + 120 days]
  resilience_loss      : area of the "resilience triangle" = sum of
                         (target - rolling FR) over days below target
Stability
  bullwhip_dc          : var(weekly store orders to DCs) / var(weekly demand)
  bullwhip_factory     : var(weekly DC orders to factory) / var(weekly demand)
"""
import numpy as np

TARGET = 0.95
WARMUP = 60


def rolling_fill_rate(sales, demand, w=7):
    k = np.ones(w)
    s = np.convolve(sales, k, "full")[:len(sales)]
    d = np.convolve(demand, k, "full")[:len(demand)]
    return np.divide(s, d, out=np.ones_like(s), where=d > 0)


def summarize(log: dict, disruption_start=None, warmup: int = WARMUP) -> dict:
    sl = slice(warmup, None)
    dem, sales = log["demand"], log["sales"]
    out = {
        "fill_rate": sales[sl].sum() / dem[sl].sum(),
        "cost_holding": log["c_hold"][sl].sum(),
        "cost_ordering": log["c_order"][sl].sum(),
        "cost_lost_sales": log["c_lost"][sl].sum(),
        "cost_backup_premium": log["c_premium"][sl].sum(),
        "avg_inventory": (log["inv_store"] + log["inv_dc"] + log["inv_fg"]
                          + log["inv_rm"])[sl].mean(),
    }
    out["cost_total"] = (out["cost_holding"] + out["cost_ordering"]
                         + out["cost_lost_sales"] + out["cost_backup_premium"])
    out["cost_operating"] = out["cost_total"] - out["cost_lost_sales"]

    # bullwhip on WEEKLY totals (daily (s,S) orders are lumpy by design)
    def wk(x):
        x = x[sl]
        n = len(x) // 7 * 7
        return x[:n].reshape(-1, 7).sum(axis=1)
    var_d = np.var(wk(dem))
    out["bullwhip_dc"] = np.var(wk(log["orders_dc"])) / var_d
    out["bullwhip_factory"] = np.var(wk(log["orders_fg"])) / var_d

    rfr = rolling_fill_rate(sales, dem)
    t0 = disruption_start if disruption_start is not None else warmup
    win = rfr[t0:]
    below = np.where(win < TARGET)[0]
    if len(below) == 0:
        out.update(time_to_impact=np.nan, recovery_time=0.0, impact_depth=1.0,
                   resilience_loss=0.0)
    else:
        drop = below[0]
        rec = np.nan
        for t in range(drop, len(win) - 14):
            if np.all(win[t:t + 14] >= TARGET):
                rec = t - drop
                break
        out.update(time_to_impact=float(drop), recovery_time=float(rec),
                   impact_depth=float(win.min()),
                   resilience_loss=float(np.clip(TARGET - win, 0, None).sum()))
    out["lost_sales_disr"] = log["lost"][t0:t0 + 120].sum()
    return out
