# 4. SVM

# pip install numpy matplotlib scikit-learn

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ---------------- DATASET ----------------

# Dataset containing 100 sample students
# Feature 1: Study Hours
# Feature 2: Attendance Percentage
# Each row represents one student's data
# Format: [Study Hours, Attendance Percentage]

X = np.array([
    # Class 0 (Fail) - First 40 students
    [1.2, 55], [1.5, 62], [1.8, 58], [2.0, 65], [2.2, 60],
    [2.4, 68], [2.6, 57], [2.8, 64], [3.0, 61], [3.1, 70],
    [1.4, 52], [1.7, 59], [2.1, 54], [2.3, 63], [2.5, 56],
    [2.7, 66], [2.9, 60], [3.2, 67], [3.3, 62], [3.5, 69],
    [1.0, 50], [1.3, 57], [1.6, 53], [1.9, 61], [2.2, 55],
    [2.5, 62], [2.8, 58], [3.0, 65], [3.4, 64], [3.6, 68],
    [1.5, 48], [1.8, 56], [2.0, 51], [2.4, 59], [2.6, 54],
    [2.9, 63], [3.1, 58], [3.3, 66], [3.5, 61], [3.7, 67],

    # Class 1 (Pass) - Remaining 60 students
    [2.8, 72], [3.0, 75], [3.2, 78], [3.4, 74], [3.6, 80],
    [3.8, 76], [4.0, 82], [4.2, 79], [4.4, 85], [4.5, 81],
    [4.7, 84], [4.9, 88], [5.0, 86], [5.2, 90], [5.4, 87],
    [5.6, 92], [5.8, 89], [6.0, 94], [6.2, 91], [6.4, 95],
    [3.9, 73], [4.1, 77], [4.3, 75], [4.5, 79], [4.7, 82],
    [4.9, 80], [5.1, 84], [5.3, 86], [5.5, 83], [5.7, 88],
    [4.0, 70], [4.2, 74], [4.4, 78], [4.6, 76], [4.8, 81],
    [5.0, 85], [5.2, 82], [5.4, 89], [5.6, 87], [5.9, 90],
    [3.1, 73], [3.5, 71], [3.8, 69], [4.0, 72], [4.3, 70],
    [4.6, 74], [4.8, 77], [5.0, 79], [5.3, 75], [5.5, 81],
    [3.2, 76], [3.6, 78], [3.9, 80], [4.1, 83], [4.4, 82],
    [4.7, 86], [5.1, 88], [5.4, 91], [5.8, 93], [6.1, 96]
])

# Target labels corresponding to the rows in X
# 0 = Class 0 (Fail)
# 1 = Class 1 (Pass)
# The labels must follow the same order as the dataset

y = np.array([
    # Labels for the first 40 students
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

    # Labels for the remaining 60 students
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1
])

# Display dataset information
print("Total data points:", len(X))
print("Number of features:", X.shape[1])

# ---------------- TRAIN-TEST SPLIT ----------------

# Split data into 80% training and 20% testing
# random_state ensures reproducible results
# stratify maintains the class proportions

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training data:", len(X_train))
print("Testing data:", len(X_test))

# ---------------- SVM MODEL ----------------

# Create a Support Vector Classifier
# Linear kernel creates a linear decision boundary
# C controls the penalty for classification errors

model = SVC(kernel="linear", C=1.0)

# Train the model
model.fit(X_train, y_train)

# Predict the classes of test data
y_pred = model.predict(X_test)

# Calculate model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nSVM Accuracy:", accuracy)

# Display classification performance
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- DATASET GRAPH ----------------

# Plot all 100 data points
# Red points represent Class 0
# Blue points represent Class 1

plt.figure(figsize=(8, 6))

plt.scatter(
    X[y == 0, 0],
    X[y == 0, 1],
    color="red",
    label="Class 0",
    s=60
)

plt.scatter(
    X[y == 1, 0],
    X[y == 1, 1],
    color="blue",
    label="Class 1",
    s=60
)

plt.xlabel("Study Hours")
plt.ylabel("Attendance Percentage")
plt.title("SVM Dataset - 100 Data Points")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

# ---------------- DECISION BOUNDARY GRAPH ----------------

# Plot the training data
# Highlight the support vectors with green circles

plt.figure(figsize=(8, 6))

plt.scatter(
    X_train[y_train == 0, 0],
    X_train[y_train == 0, 1],
    color="red",
    label="Class 0 (Train)",
    s=50
)

plt.scatter(
    X_train[y_train == 1, 0],
    X_train[y_train == 1, 1],
    color="blue",
    label="Class 1 (Train)",
    s=50
)

# Display the support vectors identified by the SVM model
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=150,
    facecolors="none",
    edgecolors="green",
    linewidths=2,
    label="Support Vectors"
)

# Define the plotting range for both features
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5

# Create a mesh grid for plotting the decision regions
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

# Predict the class for every point in the mesh grid
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Shade the two classification regions
plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.15,
    cmap=plt.cm.coolwarm
)

# Calculate the decision function values
decision = model.decision_function(
    np.c_[xx.ravel(), yy.ravel()]
)

decision = decision.reshape(xx.shape)

# Plot the decision boundary and margin lines
# 0 = Decision boundary
# -1 and +1 = Margin boundaries

plt.contour(
    xx,
    yy,
    decision,
    levels=[-1, 0, 1],
    colors="black",
    linestyles=["--", "-", "--"],
    linewidths=[1, 2, 1]
)

plt.xlabel("Study Hours")
plt.ylabel("Attendance Percentage")
plt.title("SVM Decision Boundary and Support Vectors")
plt.legend()
plt.grid(alpha=0.3)

plt.show()