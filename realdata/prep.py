"""Prepare real-data inputs for the SC-Twin proof of concept.

Reads the raw public datasets and writes compact, PII-free files to poc/data/:
  demand.npz        daily demand per (market store, SKU) from DataCo orders
  leadtimes.json    empirical lead-time distributions (DataCo, Olist, SCMS)
  meta.json         SKU / market names, dates
Run: python3 -I poc/prep.py <raw_dir> <out_dir>
"""
import sys, json
import numpy as np
import pandas as pd

RAW, OUT = sys.argv[1], sys.argv[2]

# ---------- DataCo: real orders -> daily demand ----------
cols = ['order date (DateOrders)', 'Order Item Quantity', 'Product Name', 'Market',
        'Order Status', 'Shipping Mode', 'Days for shipping (real)',
        'Days for shipment (scheduled)', 'Late_delivery_risk', 'Customer State']
dc = pd.read_csv(f'{RAW}/dataco/DataCoSupplyChainDataset.csv', encoding='latin1', usecols=cols)
dc['date'] = pd.to_datetime(dc['order date (DateOrders)']).dt.normalize()
# keep the dense period (volume drops ~80% after Sep 2017) and real demand only
dc = dc[(dc.date >= '2015-01-01') & (dc.date <= '2017-09-30')]
demand_rows = dc[~dc['Order Status'].isin(['CANCELED', 'SUSPECTED_FRAUD'])]
top = (demand_rows.groupby('Product Name')['Order Item Quantity'].sum()
       .sort_values(ascending=False).head(5).index.tolist())
# NOTE: DataCo's 'Market'/'Order Region' columns rotate in time blocks (a data artefact), so stores are
# built from the customer's state instead, which is stable across all 33 months.
STATES = {'PR': 'Puerto Rico', 'CA': 'California', 'NY': 'New York', 'TX': 'Texas'}
dc['Market'] = dc['Customer State'].map(STATES).fillna('Rest of US')
demand_rows = dc[~dc['Order Status'].isin(['CANCELED', 'SUSPECTED_FRAUD'])]
markets = ['Puerto Rico', 'California', 'New York', 'Texas', 'Rest of US']
dates = pd.date_range('2015-01-01', '2017-09-30', freq='D')
D = np.zeros((len(top), len(markets), len(dates)), dtype=np.float32)
g = (demand_rows[demand_rows['Product Name'].isin(top)]
     .groupby(['Product Name', 'Market', 'date'])['Order Item Quantity'].sum())
for (p, m, d), q in g.items():
    D[top.index(p), markets.index(m), dates.get_loc(d)] = q
np.savez_compressed(f'{OUT}/demand.npz', D=D)

# DataCo shipping days (real) per market, Standard Class -> DC-to-store transit
std = dc[dc['Shipping Mode'] == 'Standard Class']
ship = {m: std.loc[std.Market == m, 'Days for shipping (real)'].astype(int).value_counts(normalize=True)
        .sort_index().to_dict() for m in markets}
dc_late = float(dc['Late_delivery_risk'].mean())
by_mode = dc.groupby('Shipping Mode').agg(real=('Days for shipping (real)', 'mean'),
                                          sched=('Days for shipment (scheduled)', 'mean'),
                                          late=('Late_delivery_risk', 'mean')).round(3)

# ---------- SCMS (USAID): supplier delivery slip vs schedule ----------
sc = pd.read_csv(f'{RAW}/scms/SCMS_Delivery_History_Dataset.csv', encoding='utf-8-sig')
for c in ['Scheduled Delivery Date', 'Delivered to Client Date']:
    sc[c] = pd.to_datetime(sc[c], format='mixed', errors='coerce')
slip = (sc['Delivered to Client Date'] - sc['Scheduled Delivery Date']).dt.days.dropna().astype(int)
slip_late = slip[slip > 0]

# ---------- Olist: purchase -> delivered (domestic parcel lead time) ----------
ol = pd.read_csv(f'{RAW}/olist/files/olist_orders_dataset.csv',
                 parse_dates=['order_purchase_timestamp', 'order_delivered_customer_date',
                              'order_estimated_delivery_date'])
ol = ol[ol.order_status == 'delivered'].dropna(subset=['order_delivered_customer_date'])
ol_lt = (ol.order_delivered_customer_date - ol.order_purchase_timestamp).dt.days
ol_lt = ol_lt[(ol_lt >= 0) & (ol_lt <= 90)].astype(int)
ol_late = float((ol.order_delivered_customer_date > ol.order_estimated_delivery_date).mean())


def dist(s):
    vc = s.value_counts(normalize=True).sort_index()
    return {'values': vc.index.tolist(), 'p': [round(x, 6) for x in vc.values.tolist()]}


lead = {
    'dataco_ship_days_by_market': {m: {'values': list(map(int, v.keys())), 'p': list(v.values())}
                                   for m, v in ship.items()},
    'scms_slip_days': dist(slip.clip(lower=0)),
    'scms_slip_late_only': dist(slip_late),
    'olist_lead_days': dist(ol_lt),
    'summary': {
        'dataco_rows': int(len(dc)), 'dataco_late_share': round(dc_late, 4),
        'dataco_by_mode': by_mode.reset_index().to_dict('records'),
        'scms_rows': int(len(sc)), 'scms_late_share': round(float((slip > 0).mean()), 4),
        'scms_slip_p90': float(slip.quantile(.9)), 'scms_slip_p99': float(slip.quantile(.99)),
        'scms_slip_max': int(slip.max()), 'scms_late_mean': round(float(slip_late.mean()), 2),
        'olist_rows': int(len(ol)), 'olist_late_share': round(ol_late, 4),
        'olist_lead_p50': float(ol_lt.median()), 'olist_lead_p95': float(ol_lt.quantile(.95)),
    }}
json.dump(lead, open(f'{OUT}/leadtimes.json', 'w'), indent=1)
json.dump({'skus': top, 'markets': markets, 'start': str(dates[0].date()), 'end': str(dates[-1].date()),
           'days': len(dates)}, open(f'{OUT}/meta.json', 'w'), indent=1)
print('SKUs', top)
print('demand total', D.sum(), 'per sku/day', D.sum(axis=(1, 2)) / len(dates))
print(json.dumps(lead['summary'], indent=1, default=str))
