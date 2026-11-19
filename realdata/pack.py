"""Build a ready-to-import data pack for the SC-Twin Platform from the real DataCo orders.
Run after sim.py: python3 -I poc/pack.py poc/data poc/out"""
import sys, json, csv, os
import numpy as np
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2]]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim.py')).read().split("if __name__")[0]
exec(src)
R = json.load(open(f'{OUT}/results.json'))
pol = np.array([R['retuned']])
P = f'{OUT}/platform_import'; os.makedirs(P, exist_ok=True)
ids = ['puerto-rico', 'california', 'new-york', 'texas', 'rest-of-us']
names = [m + ' store' for m in META['markets']]
skus = ['rip-deck', 'golf-polo', 'life-vest', 'running-shoe', 'toddler-shoe']
price = {'rip-deck': 59.99, 'golf-polo': 50.0, 'life-vest': 49.98, 'running-shoe': 99.99, 'toddler-shoe': 39.99}
start = np.datetime64(META['start'], 'D')
w = lambda f, rows: csv.writer(open(f'{P}/{f}', 'w', newline='')).writerows(rows)
mean_lt = [float(np.dot(*STORE_LT[j])) for j in range(NJ)]
last28 = D[:, :, -28:].sum(0).mean(1)
w('1_locations.csv', [['id', 'name', 'type', 'source', 'backup', 'lead_days', 'demand_per_day', 'capacity_per_day'],
                      ['supplier', 'Main supplier', 'supplier', '', '', '', '', ''],
                      ['central-dc', 'Central DC', 'dc', 'supplier', '', QUOTED, round(float(last28.sum()), 1), '']] +
  [[ids[j], names[j], 'store', 'central-dc', '', round(mean_lt[j], 1), round(float(last28[j]), 1), ''] for j in range(NJ)])
w('2_products.csv', [['sku', 'name', 'unit', 'value']] + [[skus[k], META['skus'][k], 'pcs', price[skus[k]]] for k in range(NSKU)])
rows = [['date', 'location', 'sku', 'qty']]
for k in range(NSKU):
    for j in range(NJ):
        for t in range(T - 120, T):
            rows.append([str(start + np.timedelta64(t, 'D')), ids[j], skus[k], int(D[k, j, t])])
w('3_sales.csv', rows)
stock = [['location', 'sku', 'on_hand']]; orders = [['no', 'from', 'to', 'sku', 'qty', 'status', 'eta']]
end = start + np.timedelta64(T - 1, 'D'); n = 1
for k in range(NSKU):
    *_, rec = simulate(pol, 'international', 1, k, record=True)
    for j in range(NJ):
        stock.append([ids[j], skus[k], int(rec['final_store'][0, j])])
    stock.append(['central-dc', skus[k], int(rec['final_dc'][0])])
    for j, dd, q in rec['open_store']:
        orders.append([f'REPLAY-{n:04d}', 'central-dc', ids[j], skus[k], int(q), 'in_transit', str(end + np.timedelta64(dd, 'D'))]); n += 1
    for dd, q in rec['open_dc']:
        orders.append([f'REPLAY-{n:04d}', 'supplier', 'central-dc', skus[k], int(q), 'in_transit', str(end + np.timedelta64(dd, 'D'))]); n += 1
w('4_stock.csv', stock); w('5_open_orders.csv', orders)
print(len(rows) - 1, 'sales rows;', len(stock) - 1, 'stock rows;', len(orders) - 1, 'open orders')
