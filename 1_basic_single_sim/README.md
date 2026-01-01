# 1_basic_single_sim

Goal: a single, reproducible simulation. No parallelism.

Run:

```bash
python run_single.py \
  --steps 10000 --p1 0.5 --p2 0.47 \
  --init-mailly 10 --init-moulin 5 \
  --seed 123 \
  --out-csv results.csv --plot
```

Outputs:
- results.csv: time series with columns: time, mailly, moulin
- mailly.png: plot of counts over time
