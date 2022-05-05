# Optimizing-ML-Query

## run

### run synthetic experiments (generation time)
```
chmod +x run/*
./run/get_synthetic_plan.sh
```

### run on real dataset (COCO, VOC)

```
./run/get_plan.sh
./run/run_coco_baseline.sh
```

### To run on specific configurations
```
# SYNTHETIC
outdir=synthetic_cost
q_file=simulation/synthetic_query_uniform.csv
outdir=synthetic_accu

bound=0.9
query_idx = 0

python3 query_optimizer.py -mdist uniform -qdist uniform -synthetic -record -qfile ${q_file} -query-idx ${query_idx} -outdir ${outdir} -constraint accuracy -bound ${bound} -order
```