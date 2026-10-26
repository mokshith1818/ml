# 6. PCA

# pip install pandas numpy matplotlib scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create an inline CSV dataset
csv_data = StringIO("""Coding_Hours,Tasks_Completed,Bugs_Fixed,Productivity_Score
5,8,3,65
6,10,4,72
7,12,5,78
4,6,2,58
8,14,6,85
3,5,1,50
9,16,7,92
5,9,3,68
7,13,5,82
6,11,4,75
10,18,8,96
4,7,2,60
8,15,6,88
3,4,1,45
9,17,7,94
""")

# Read the inline CSV data
df = pd.read_csv(csv_data)

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate individual and cumulative explained variance
exp_var_ratio = pca.explained_variance_ratio_ * 100
cum_var_ratio = np.cumsum(exp_var_ratio)

components = [f'PC_{i+1}' for i in range(len(exp_var_ratio))]

# Display explained variance table
print("\n" + "=" * 55)
print(f"{'Component':<12} | {'Individual Var (%)':<20} | {'Cumulative Var (%)':<20}")
print("=" * 55)

for i in range(len(components)):
    print(f"{components[i]:<12} | {exp_var_ratio[i]:<20.2f} | {cum_var_ratio[i]:<20.2f}")

print("=" * 55 + "\n")

# Plot individual and cumulative explained variance
plt.figure(figsize=(9, 5))

bars = plt.bar(
    components,
    exp_var_ratio,
    color='royalblue',
    alpha=0.7,
    label='Individual Variance (%)'
)

plt.plot(
    components,
    cum_var_ratio,
    color='firebrick',
    marker='o',
    linewidth=2,
    label='Cumulative Variance (%)'
)

# Display variance values above the bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2.,
        height + 1,
        f'{height:.1f}%',
        ha='center',
        va='bottom'
    )

plt.title('PCA Explained Variance for Developer Metrics', fontsize=14, fontweight='bold')
plt.xlabel('Principal Components', fontsize=12)
plt.ylabel('Explained Variance (%)', fontsize=12)
plt.ylim(0, 110)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend(loc='center right')
plt.tight_layout()
plt.show()