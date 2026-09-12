import pandas as pd

data = pd.read_csv("heart.csv")

print("Target Counts:")
print(data["target"].value_counts())

print("\nSample rows with target = 0")
print(data[data["target"] == 0].head(3))

print("\nSample rows with target = 1")
print(data[data["target"] == 1].head(3))