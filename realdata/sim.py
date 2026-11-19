"""SC-Twin proof of concept: replay REAL orders through the twin's reorder logic.

Network  : supplier -> central DC -> 5 regional stores (Puerto Rico, California, New York, Texas, rest of US)
Demand   : DataCo daily order quantities, 5 top SKUs, 2015-01-01 .. 2017-09-30 (real, replayed as-is)
DC->store: DataCo real 'Days for shipping' (Standard Class) per region
Supplier->DC lead time, three profiles built from real deliveries:
  domestic      Olist purchase->delivered days (Brazil, 96k parcels)
  international 14-day quoted lead + USAID SCMS slip vs schedule (10k shipments)
  disrupted     international, plus a 60-day window where every shipment draws from SCMS *late* shipments only
Policy   : (s, S) in days of cover, exactly as in the SC-Twin Platform (forecast = trailing 28-day mean)
Monte Carlo over lead-time draws; demand is fixed real history.
Run: python3 -I poc/sim.py poc/data poc/out
"""
import sys, json, time
import numpy as np

DATA, OUT = sys.argv[1], sys.argv[2]
D = np.load(f'{DATA}/demand.npz')['D'].astype(np.float64)   # [sku, store, day]
LT = json.load(open(f'{DATA}/leadtimes.json'))
META = json.load(open(f'{DATA}/meta.json'))
NSKU, NJ, T = D.shape
WARM = 60
QUOTED = 14
DIS_START, DIS_LEN = 790, 60          # 2017-03-01 .. 2017-04-29, inside the 2017 test year
TRAIN_END = 731                       # 2015-01-01 .. 2016-12-31 calibration; 2017 = test

KNEE = dict(store=(5.9, 13.3), dc=(8.3, 15.0))
LEAN = dict(store=(2.5, 5.0), dc=(5.0, 9.0))
BUFF = dict(store=(4.0, 8.0), dc=(9.0, 16.0))


def sampler(d):
    v = np.array(d['values'], dtype=np.int64); p = np.array(d['p'], dtype=np.float64)
    return v, p / p.sum()


STORE_LT = [sampler(LT['dataco_ship_days_by_market'][m]) for m in META['markets']]
OLIST = sampler(LT['olist_lead_days'])
SLIP = sampler(LT['scms_slip_days'])
SLIP_LATE = sampler(LT['scms_slip_late_only'])


def dc_lead(profile, rng, n, t):
    if profile == 'domestic':
        return np.maximum(1, rng.choice(OLIST[0], n, p=OLIST[1]))
    s = rng.choice(SLIP[0], n, p=SLIP[1])
    if profile == 'disrupted' and DIS_START <= t < DIS_START + DIS_LEN:
        s = rng.choice(SLIP_LATE[0], n, p=SLIP_LATE[1])
    return QUOTED + s


def simulate(pols, profile, seeds, sku, record=False):
    """pols: array [P,4] = store_s, store_S, dc_s, dc_S (days). Returns per-(policy,seed) daily KPIs."""
    P = len(pols); B = P * seeds
    ss = np.repeat(pols[:, 0], seeds)[:, None]; SS = np.repeat(pols[:, 1], seeds)[:, None]
    ds = np.repeat(pols[:, 2], seeds); DS = np.repeat(pols[:, 3], seeds)
    rng = np.random.default_rng(1000 + sku)
    dem = D[sku]                                         # [J, T]
    cs = np.concatenate([np.zeros((NJ, 1)), np.cumsum(dem, axis=1)], axis=1)
    H = T + 260
    arr_s = np.zeros((B, NJ, H)); arr_d = np.zeros((B, H))
    f0 = dem[:, :28].mean(axis=1)
    on_s = SS * f0[None, :]; oo_s = np.zeros((B, NJ))
    on_d = DS * f0.sum(); oo_d = np.zeros(B)
    served = np.zeros((B, T)); demand = np.zeros(T); inv = np.zeros((B, T)); so = np.zeros((B, T))
    rec = {'store_on': [], 'dc_on': [], 'lost': []} if record else None
    for t in range(T):
        on_s += arr_s[:, :, t]; oo_s -= arr_s[:, :, t]
        on_d += arr_d[:, t]; oo_d -= arr_d[:, t]
        d = dem[:, t][None, :]
        sv = np.minimum(on_s, d); on_s -= sv
        served[:, t] = sv.sum(1); demand[t] = d.sum()
        so[:, t] = ((sv < d) & (d > 0)).sum(1)
        lo = max(0, t - 27)
        f = (cs[:, t + 1] - cs[:, lo]) / (t + 1 - lo)       # trailing 28-day mean, per store
        # store replenishment from DC
        ip = on_s + oo_s
        q = np.where(ip <= ss * f[None, :], SS * f[None, :] - ip, 0.0)
        need = q.sum(1)
        frac = np.where(need > on_d, on_d / np.maximum(need, 1e-9), 1.0)
        ship = q * frac[:, None]
        on_d -= ship.sum(1)
        for j in range(NJ):
            L = np.maximum(1, rng.choice(STORE_LT[j][0], B, p=STORE_LT[j][1]))
            idx = np.minimum(t + L, H - 1)
            np.add.at(arr_s[:, j, :], (np.arange(B), idx), ship[:, j])
        oo_s += ship
        # DC replenishment from supplier
        fd = f.sum()
        ipd = on_d + oo_d
        qd = np.where(ipd <= ds * fd, DS * fd - ipd, 0.0)
        L = dc_lead(profile, rng, B, t)
        idx = np.minimum(t + L, H - 1)
        np.add.at(arr_d, (np.arange(B), idx), qd)
        oo_d += qd
        inv[:, t] = on_s.sum(1) + on_d
        if record:
            rec['store_on'].append(on_s.sum(1).copy()); rec['dc_on'].append(on_d.copy())
            rec['lost'].append((d - sv).sum(1).copy())
    if record:
        rec['final_store'] = on_s.copy(); rec['final_dc'] = on_d.copy()
        rec['open_store'] = [(j, int(t2 - (T - 1)), float(arr_s[0, j, t2])) for j in range(NJ) for t2 in range(T, H) if arr_s[0, j, t2] > 0.5]
        rec['open_dc'] = [(int(t2 - (T - 1)), float(arr_d[0, t2])) for t2 in range(T, H) if arr_d[0, t2] > 0.5]
    return served.reshape(P, seeds, T), demand, inv.reshape(P, seeds, T), so.reshape(P, seeds, T), rec


def kpis(served, demand, inv, so, a, b):
    sv = served[..., a:b].sum(-1); dm = demand[a:b].sum()
    return dict(fill=sv / dm, inv_days=inv[..., a:b].mean(-1) / (dm / (b - a)),
                stockout_days=(so[..., a:b] > 0).mean(-1))


def run_all(pols, profile, seeds):
    """Aggregate across the 5 SKUs. Returns dict of arrays [P, seeds] for train and test windows."""
    acc = {w: {'sv': 0, 'dm': 0, 'inv': 0, 'so': 0} for w in ('train', 'test', 'all')}
    win = {'train': (WARM, TRAIN_END), 'test': (TRAIN_END, T), 'all': (WARM, T)}
    for k in range(NSKU):
        sv, dm, inv, so, _ = simulate(pols, profile, seeds, k)
        for w, (a, b) in win.items():
            acc[w]['sv'] = acc[w]['sv'] + sv[..., a:b].sum(-1)
            acc[w]['dm'] = acc[w]['dm'] + dm[a:b].sum()
            acc[w]['inv'] = acc[w]['inv'] + inv[..., a:b].mean(-1)
            acc[w]['so'] = acc[w]['so'] + (so[..., a:b] > 0).mean(-1) / NSKU
    out = {}
    for w, (a, b) in win.items():
        x = acc[w]; daily = x['dm'] / (b - a)
        out[w] = dict(fill=x['sv'] / x['dm'], inv_days=x['inv'] / daily, stockout=x['so'])
    return out


if __name__ == '__main__':
    import os; os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    # 1) calibrate SC-Twin's knee policy on REAL 2015-16 data (international profile)
    grid = np.array([(s, s + a, ds, ds + b) for s in (3, 5, 7, 9, 11, 13) for a in (3, 6, 9)
                     for ds in (6, 10, 14, 18, 22, 26, 30) for b in (4, 8, 12, 18)], dtype=float)
    res = run_all(grid, 'international', seeds=8)
    fill = res['train']['fill'].mean(1); invd = res['train']['inv_days'].mean(1)
    # knee = least stock that keeps fill >= 98% (or within 1 point of the best, if 98% is unreachable)
    target = min(0.98, fill.max() - 0.01)
    ok = np.where(fill >= target)[0]
    best = ok[np.argmin(invd[ok])]
    RETUNED = grid[best]
    frontier = [dict(pol=grid[i].tolist(), fill=round(float(fill[i]), 4), inv_days=round(float(invd[i]), 2))
                for i in range(len(grid))]
    print('grid done', round(time.time() - t0), 's; retuned', RETUNED, fill[best], invd[best], 'target', target)

    # 2) evaluate all policies on every lead-time profile, 2017 = out-of-sample
    named = {'SC-Twin v1.0 (synthetic-tuned)': KNEE, 'Lean': LEAN, 'Buffered': BUFF}
    pols = np.array([[*v['store'], *v['dc']] for v in named.values()] + [RETUNED.tolist()])
    names = list(named) + ['SC-Twin re-tuned on real data']
    table = {}
    for prof in ('domestic', 'international', 'disrupted'):
        r = run_all(pols, prof, seeds=40)
        table[prof] = {n: {w: {k: [round(float(r[w][k][i].mean()), 4), round(float(np.percentile(r[w][k][i], 5)), 4),
                                    round(float(np.percentile(r[w][k][i], 95)), 4)] for k in r[w]}
                           for w in r} for i, n in enumerate(names)}
        print(prof, {n: (table[prof][n]['test']['fill'][0], table[prof][n]['test']['inv_days'][0]) for n in names})

    # 3) one replay trace for the page: top SKU, disrupted profile, seed 0, all four policies
    trace = {}
    for i, n in enumerate(names):
        sv, dm, inv, so, rec = simulate(pols[i:i + 1], 'disrupted', 1, 0, record=True)
        trace[n] = dict(store=[round(float(x[0])) for x in rec['store_on']],
                        dc=[round(float(x[0])) for x in rec['dc_on']],
                        lost=[round(float(x[0]), 1) for x in rec['lost']])
    json.dump(dict(meta=META, policies=dict(zip(names, pols.tolist())), retuned=RETUNED.tolist(),
                   target_fill=round(float(target), 4), frontier=frontier, table=table,
                   demand_top=[round(float(x)) for x in D[0].sum(0)], trace=trace,
                   disruption=dict(start=DIS_START, length=DIS_LEN), train_end=TRAIN_END, warm=WARM,
                   quoted_lead=QUOTED, leadsummary=LT['summary']),
              open(f'{OUT}/results.json', 'w'))
    print('done', round(time.time() - t0), 's')
