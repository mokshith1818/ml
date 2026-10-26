# 1.A. Linear Regression

# pip install pandas matplotlib scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample dataset (Inline)
data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [20, 30, 46, 50, 65, 70, 80, 90, 95, 100]
})

print("Dataset:\n")
print(data)

X = data[["Hours"]]
y = data["Marks"]

model = LinearRegression()

# Train model
model.fit(X, y)
predictions = model.predict(X)

# Display equation
print("\nSlope (m):", model.coef_[0])
print("Intercept (b):", model.intercept_)
print(f"\nRegression Equation:")
print(f"Marks = {model.coef_[0]:.2f} × Hours + {model.intercept_:.2f}")

new_hours = pd.DataFrame({"Hours": [7.5]})
predicted_marks = model.predict(new_hours)

print(f"\nPredicted Marks for 7.5 Hours = {predicted_marks[0]:.2f}")

# Plot graph
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, predictions, color="red", linewidth=2, label="Regression Line")
plt.title("Linear Regression")
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.legend()
plt.grid(True)

plt.show()