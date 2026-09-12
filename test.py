import pandas as pd
import joblib

data = pd.read_csv("heart.csv")
model = joblib.load("heart_model.pkl")

for i in range(10):
    row = data.iloc[i]

    X = pd.DataFrame(
        [row.drop("target").values],
        columns=data.drop("target", axis=1).columns
    )

    pred = model.predict(X)[0]

    print(
        "Row:", i,
        "| Actual:", row["target"],
        "| Predicted:", pred
    )