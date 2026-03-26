# Reproducibility Notes

## Dataset
- Source: Breast Cancer Wisconsin (Diagnostic)
- Local file: `data/data.csv`
- Samples: 569
- Predictive features after cleaning: 30
- Target: `diagnosis` mapped from `{M, B}` to `{1, 0}`

## Cleaning steps
1. Drop `Unnamed: 32`
2. Drop `id`
3. Map labels `M -> 1`, `B -> 0`

## Training setup
- Model: `LogisticRegression`
- Feature scaling: `StandardScaler`
- Split: `train_test_split(test_size=0.2, random_state=42)`
- Environment: CPU only

## Reproducibility advice
To make the project easier to reproduce, keep the following committed:
- `requirements.txt`
- evaluation logs
- saved metrics JSON/CSV
- test files
- a project license

## Example command
```bash
python scripts/evaluate_model.py
pytest tests -q
```
