# 3.B. Ensemble Technique - Boosting

# pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

X = np.array([
    [1.2, 62], [1.5, 68], [1.8, 55], [2.0, 72], [2.2, 64],
    [2.4, 58], [2.6, 70], [2.8, 61], [3.0, 75], [3.1, 66],
    [3.3, 69], [3.5, 78], [3.6, 63], [3.8, 71], [4.0, 80],
    [4.1, 74], [4.3, 82], [4.5, 68], [4.6, 85], [4.8, 77],
    [5.0, 88], [5.1, 73], [5.2, 91], [5.4, 79], [5.5, 86],
    [5.7, 94], [5.8, 81], [6.0, 89], [6.1, 76], [6.3, 93],
    [6.4, 84], [6.6, 96], [6.7, 88], [6.9, 91], [7.0, 82],
    [7.2, 95], [7.3, 87], [7.5, 92], [7.6, 80], [7.8, 97],
    [2.7, 82], [3.2, 54], [4.2, 59], [4.7, 62], [5.3, 67],
    [5.9, 72], [6.2, 65], [6.8, 74], [3.9, 57], [7.4, 78]
])

y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    1, 1, 0, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    0, 0, 0, 1, 1,
    1, 1, 1, 0, 1
])

print("========================================")
print("ENSEMBLE LEARNING - BOOSTING")
print("========================================")

print("\nTotal Data Points:", len(X))
print("Features: Study Hours, Attendance")
print("Target: 0 = Fail, 1 = Pass")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ---------------- BOOSTING ----------------

boosting_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

boosting_model.fit(X_train, y_train)

boosting_pred = boosting_model.predict(X_test)
boosting_accuracy = accuracy_score(y_test, boosting_pred)

print("\n========== BOOSTING ==========")

print("Boosting Accuracy:", round(boosting_accuracy * 100, 2), "%")

print("\nBoosting Classification Report:")
print(classification_report(
    y_test,
    boosting_pred,
    target_names=["Fail", "Pass"]
))

# ---------------- CONFUSION MATRIX ----------------

plt.figure(figsize=(7, 6))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    boosting_pred,
    display_labels=["Fail", "Pass"],
    cmap="Greens"
)

plt.title("Boosting - Confusion Matrix")
plt.show()