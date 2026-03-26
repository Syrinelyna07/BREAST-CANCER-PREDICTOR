# Biases and Limitations

## Known limitations
- The dataset is relatively small (569 samples).
- It is a classic benchmark dataset and may not reflect broader clinical populations.
- This project is a machine-learning demo and not a validated clinical decision system.
- Accuracy alone is not sufficient in a medical setting because false negatives are especially costly.

## Practical risk
A malignant case close to the decision boundary could be classified as benign. This is why recall for the malignant class should be monitored carefully, not only overall accuracy.

## Recommended next steps
- Add stratified cross-validation
- Inspect false negatives explicitly
- Add calibration checks
- Document training data provenance and intended use
- Keep the warning in the app that this does not replace medical advice
