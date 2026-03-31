# src/models/evaluate.py
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test, scaler=None):
    if scaler:
        X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return y_prob