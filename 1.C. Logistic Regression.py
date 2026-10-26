# 1.C. Logistic Regression

# pip install numpy scikit-learn matplotlib

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Sample dataset
# Hours Studied
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])

# Result (0 = Fail, 1 = Pass)
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Take user input
hours = float(input("Enter the number of hours studied: "))

# Predict result
prediction = model.predict([[hours]])
probability = model.predict_proba([[hours]])

# Display result
if prediction[0] == 1:
    print("Prediction: PASS")
else:
    print("Prediction: FAIL")

print("Probability of Passing:", round(probability[0][1] * 100, 2), "%")

# Plot the dataset
plt.scatter(X, y, color="blue", label="Training Data")

# Logistic curve
X_test = np.linspace(1, 10, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]

plt.plot(X_test, y_prob, color="red", label="Logistic Curve")

# User input point
plt.scatter(
    hours,
    probability[0][1],
    color="green",
    s=100,
    label="Prediction"
)

plt.title("Logistic Regression")
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.yticks([0, 1], ["Fail", "Pass"])
plt.grid(True)
plt.legend()
plt.show()