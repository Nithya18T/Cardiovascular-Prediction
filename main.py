import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("heart.csv")

X = data.drop("target", axis=1)
y = data["target"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# Gradient Boosting
gb = GradientBoostingClassifier(
    random_state=42
)


# Soft Voting Ensemble
ensemble = VotingClassifier(
    estimators=[
        ("rf", rf),
        ("gb", gb)
    ],
    voting="soft"
)


# Train ensemble
ensemble.fit(X_train, y_train)


# Prediction
pred = ensemble.predict(X_test)


# Accuracy
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", round(accuracy * 100, 2), "%")


# Save model
joblib.dump(
    ensemble,
    "heart_model.pkl"
)

print("heart_model.pkl saved successfully.")


# Feature importance
rf.fit(X_train, y_train)

feature_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

feature_df.to_csv(
    "feature_importance.csv",
    index=False
)

print("feature_importance.csv saved successfully.")