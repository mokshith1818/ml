# 2.B. CART - Regression

# pip install scikit-learn matplotlib

from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

hours = [
    1.0, 1.2, 1.4, 1.5, 1.7, 1.8, 2.0, 2.1, 2.2, 2.4,
    2.5, 2.6, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5,
    3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5,
    4.6, 4.7, 4.8, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6,
    5.7, 5.8, 6.0, 6.1, 6.2, 6.3, 6.5, 6.6, 6.8, 7.0
]

marks = [
    35, 38, 40, 42, 43, 45, 46, 48, 49, 50,
    52, 53, 55, 56, 57, 58, 59, 60, 61, 62,
    63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
    83, 84, 85, 86, 87, 88, 89, 90, 92, 94
]

X = [[h] for h in hours]

# ---------------- REGRESSION ----------------

reg = DecisionTreeRegressor(
    criterion="squared_error",
    max_depth=3
)

reg.fit(X, marks)

# ---------------- USER INPUT ----------------

study_hours = float(input("Enter study hours: "))

# Regression prediction
marks_prediction = reg.predict([[study_hours]])

print("\n----- Prediction Result -----")
print("Study Hours:", study_hours)
print("Predicted Marks:", round(marks_prediction[0], 2))

# ---------------- REGRESSION GRAPH ----------------

fig2 = plt.figure(figsize=(12, 7))

fig2.canvas.manager.set_window_title(
    "ML Exp2 - Figure 2 - Cart Regression Tree (moba)"
)

plot_tree(
    reg,
    feature_names=["Study Hours"],
    filled=True,
    fontsize=8
)

plt.title("CART Regression Tree", fontsize=14)
plt.show()