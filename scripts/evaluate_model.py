from pathlib import Path
import json
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "repo_snapshot" / "data.csv"
OUT_DIR = ROOT / "data_out"
OUT_DIR.mkdir(exist_ok=True)


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

    X = data.drop(["diagnosis"], axis=1)
    y = data["diagnosis"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset": "Breast Cancer Wisconsin (Diagnostic)",
        "samples": int(len(data)),
        "features": int(X.shape[1]),
        "split": "80/20 random split",
        "random_state": 42,
        "model": "LogisticRegression",
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_malignant": precision_score(y_test, y_pred),
        "recall_malignant": recall_score(y_test, y_pred),
        "f1_malignant": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    with open(OUT_DIR / "baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(OUT_DIR / "final_validation_log.txt", "w", encoding="utf-8") as f:
        f.write("Final validation summary\n")
        f.write("=" * 24 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision (malignant): {metrics['precision_malignant']:.4f}\n")
        f.write(f"Recall (malignant): {metrics['recall_malignant']:.4f}\n")
        f.write(f"F1 (malignant): {metrics['f1_malignant']:.4f}\n")
        f.write(f"ROC-AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"Confusion Matrix: {metrics['confusion_matrix']}\n\n")
        f.write(classification_report(y_test, y_pred))

    with open(OUT_DIR / "baseline_run.csv", "w", encoding="utf-8") as f:
        f.write(
            "run_name,model,scaler,split,random_state,accuracy,precision_malignant,recall_malignant,f1_malignant,roc_auc\n"
        )
        f.write(
            f"baseline_logreg,LogisticRegression,StandardScaler,0.2,42,{metrics['accuracy']:.6f},{metrics['precision_malignant']:.6f},{metrics['recall_malignant']:.6f},{metrics['f1_malignant']:.6f},{metrics['roc_auc']:.6f}\n"
        )

    with open(OUT_DIR / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(OUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()
