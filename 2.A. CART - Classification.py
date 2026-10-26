# 2.A. CART - Classification

# pip install scikit-learn matplotlib

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

hours = [
    1.0, 1.2, 1.4, 1.5, 1.7, 1.8, 2.0, 2.1, 2.2, 2.4,
    2.5, 2.6, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5,
    3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5,
    4.6, 4.7, 4.8, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6,
    5.7, 5.8, 6.0, 6.1, 6.2, 6.3, 6.5, 6.6, 6.8, 7.0
]

# Classification: 0 = Fail, 1 = Pass
result = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
]

X = [[h] for h in hours]

# ---------------- CLASSIFICATION ----------------

clf = DecisionTreeClassifier(
    criterion="gini",
    max_depth=3
)

clf.fit(X, result)

# ---------------- USER INPUT ----------------

study_hours = float(input("Enter study hours: "))

# Classification prediction
classification_prediction = clf.predict([[study_hours]])

if classification_prediction[0] == 1:
    result_text = "Pass"
else:
    result_text = "Fail"

print("\n----- Prediction Result -----")
print("Study Hours:", study_hours)
print("Classification Prediction:", result_text)

# ---------------- CLASSIFICATION GRAPH ----------------

fig1 = plt.figure(figsize=(12, 8))

fig1.canvas.manager.set_window_title(
    "ML Exp2 - Figure 1 - Cart Classification Tree (moba)"
)

plot_tree(
    clf,
    feature_names=["Study Hours"],
    class_names=["Fail", "Pass"],
    filled=True,
    fontsize=14
)

plt.title("CART Classification Tree", fontsize=14)
plt.show()