import pandas as pd
import os
import numpy as np

results_path = os.path.join('.', "results/iDLG_lenet_cifar10/results_fo_cpu.csv")
df = pd.read_csv(results_path, header=None)

# df = df.iloc[0:200, :] # use only 200 samples
thresholds = [0.01, 0.005, 0.001, 0.0005, 0.0001]
for threshold in thresholds:
    print(f"MSE < {threshold}:", len(df[df.iloc[:, 2] < threshold]) / len(df)) # MSE threshold

loss = df.iloc[:, 1].mean()
mse = df.iloc[:, 2].mean()

lossm = df.iloc[:, 1].median()
msem = df.iloc[:, 2].median()

min_index = df.iloc[:, 2].idxmin()

print(f"Mean Loss: {loss:.4e}, Mean MSE: {mse:.4e}")
print(f"Median Loss: {lossm:.4e}, Median MSE: {msem:.4e}")
print(f"Minimum MSE: {df.iloc[min_index, 2]}, Index: {df.iloc[min_index, 0]}")