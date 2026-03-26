from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "repo_snapshot" / "data.csv"


def get_clean_data_for_test() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data = data.drop(["Unnamed: 32", "id"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


def test_clean_data_shape_and_columns():
    data = get_clean_data_for_test()
    assert "id" not in data.columns
    assert "Unnamed: 32" not in data.columns
    assert "diagnosis" in data.columns
    assert data.shape[1] == 31


def test_labels_are_binary():
    data = get_clean_data_for_test()
    assert set(data["diagnosis"].unique()) == {0, 1}
