import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

X = [[1],[2],[3],[4],[5],[6],[7]]
y = [0,0,0,0,1,1,1]

m = DecisionTreeClassifier(max_depth=3).fit(X,y)
h = float(input("Hours: "))
print("Pass" if m.predict([[h]])[0] else "Fail")

plot_tree(m,filled=True)
plt.show()
