# 1.B. Multivariate Linear Regression

# pip install numpy scikit-learn matplotlib

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dataset
# [Study Hours, Attendance (%), Assignment Score]
X = np.array([
    [2, 65, 50],
    [3, 70, 55],
    [4, 72, 60],
    [5, 75, 65],
    [6, 80, 70],
    [7, 82, 75],
    [8, 85, 80],
    [9, 90, 85],
    [10, 92, 88],
    [3, 68, 52],
    [4, 74, 62],
    [5, 78, 68],
    [6, 81, 73],
    [7, 84, 77],
    [8, 87, 82],
    [9, 89, 86],
    [10, 94, 92],
    [4, 76, 64],
    [6, 83, 71],
    [8, 88, 84]
])

# Final Marks
y = np.array([
    48, 55, 61, 67, 74, 79, 85, 91, 95, 51,
    64, 70, 76, 81, 86, 90, 97, 66, 75, 88
])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Prediction for training data
y_pred = model.predict(X)

# User Input
study_hours = float(input("\nEnter Study Hours: "))
attendance = float(input("Enter Attendance (%): "))
assignment_score = float(input("Enter Assignment Score: "))

# Predict final marks
marks = model.predict([[study_hours, attendance, assignment_score]])

print("\nPredicted Final Marks:", round(marks[0], 2))

# Graph: Actual vs Predicted Values
plt.figure(figsize=(8, 6))

# Actual vs Predicted values
plt.scatter(
    y_pred,
    y,
    color="blue",
    s=70,
    label="Actual vs Predicted"
)

# Ideal prediction line
min_value = min(y.min(), y_pred.min())
max_value = max(y.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    color="red",
    linewidth=2,
    label="Ideal Prediction"
)

plt.title("Multivariate Linear Regression")
plt.xlabel("Predicted Marks")
plt.ylabel("Actual Marks")

plt.grid(True)
plt.legend()

plt.show()