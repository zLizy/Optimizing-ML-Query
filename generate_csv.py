#!/usr/bin/python3.8

import os, json
import pandas as pd

df  = pd.DataFrame()
experiments = os.listdir('./results/script')
print(experiments)
for e in experiments:
    workflows = os.listdir(f"./results/script/{e}/")
    for w in workflows:
        f = open(f"./results/script/{e}/{w}")
        metrics = json.load(f)
        row = {
            "experiment": e,
            "workflow": metrics['metadata']['name'],
            "startedAt": metrics['status']['startedAt'],
            "finishedAt": metrics['status']['finishedAt'],
            "resourcesDuration_cpu": metrics['status']['resourcesDuration']['cpu'],
            "resourcesDuration_memory": metrics['status']['resourcesDuration']['memory'],
        }
        df = df.append(row, ignore_index=True)

df['duration'] = pd.to_datetime(df['finishedAt'], infer_datetime_format=True) - pd.to_datetime(df['startedAt'], infer_datetime_format=True)
df['duration'] = df['duration'].dt.seconds
df = df[["experiment", "workflow", "startedAt", "finishedAt", "duration", "resourcesDuration_cpu", "resourcesDuration_memory"]]
df.to_csv("./results/results.csv",index=False)
