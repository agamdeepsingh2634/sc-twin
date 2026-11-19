# Real-data replay (proof of concept, future scope)

SC-Twin v1.0 was built and tuned on synthetic demand. This folder replays **real public records** through the same
(s, S) days-of-cover reorder logic used by the SC-Twin Platform.

| Dataset | What it supplies | Rows | Licence | Get it |
|---|---|---|---|---|
| DataCo Smart Supply Chain (Constante, Silva & Pereira) | Daily demand (5 top SKUs, 2015-01 to 2017-09) and real DC-to-store shipping days | 171,962 order lines used | CC BY 4.0 | Mendeley Data, doi:10.17632/8gx2fvg2k6 |
| Olist Brazilian E-Commerce | Domestic supplier lead-time profile (purchase to delivery) | 96,470 delivered orders | CC BY-NC-SA 4.0 | Kaggle: olistbr/brazilian-ecommerce |
| USAID SCMS Delivery History | Supplier slips vs schedule, drives the disruption scenario | 10,324 shipments | US public domain | data.gov: Supply Chain Shipment Pricing Data |

Raw files are not committed (size and licence). Put them under `datasets/raw/` as
`dataco/DataCoSupplyChainDataset.csv`, `olist/files/olist_orders_dataset.csv`, `scms/SCMS_Delivery_History_Dataset.csv`.

```
python prep.py datasets/raw data   # demand + empirical lead-time distributions (no personal fields kept)
python sim.py  data out            # calibrate on 2015-16, test on 2017, three lead-time profiles
python pack.py data out            # CSVs in the SC-Twin Platform import format
```

## Results (2017, out-of-sample, mean of 40 lead-time draws)

| Policy (store s/S, DC s/S, days) | Domestic fill | International fill | Disrupted fill | Stock cover (intl.) |
|---|---|---|---|---|
| Re-tuned on real data (9/12, 18/22) | 99.2% | 97.9% | 89.1% | 11.2 d |
| v1.0 as submitted (5.9/13.3, 8.3/15) | 84.8% | 77.3% | 68.0% | 5.3 d |
| Buffered (4/8, 9/16) | 77.5% | 72.2% | 63.5% | 4.1 d |
| Lean (2.5/5, 5/9) | 52.4% | 43.2% | 37.4% | 1.5 d |

The knee-point method transfers; the v1.0 numbers do not, because real suppliers quote 14 days and slip on ~11% of
shipments, so the DC reorder point must exceed the real lead time.

Notes: DataCo's `Market` column rotates between regions in blocks of months (dataset artefact), so stores are grouped by
customer state. The datasets come from different companies, so the network is assembled, not one firm's chain.
