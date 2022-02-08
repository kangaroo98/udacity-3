import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("./fake_data.csv")

X = df["feature"].values
y = df["label"].values

print(f"Before: {X}")
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, 1))
print(f"After: {X}")

np.savetxt("X.csv", X)
np.savetxt("y.csv", y)