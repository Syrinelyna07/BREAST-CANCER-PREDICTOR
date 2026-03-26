from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "repo_snapshot" / "data.csv"


def test_training_pipeline_reaches_reasonable_accuracy():
    data = pd.read_csv(DATA_PATH)
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

    X = data.drop(["diagnosis"], axis=1)
    y = data["diagnosis"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert accuracy_score(y_test, y_pred) >= 0.95
