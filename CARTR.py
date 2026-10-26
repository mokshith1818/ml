import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

X = [[1],[2],[3],[4],[5],[6],[7]]
y = [35,40,50,60,70,80,90]

m = DecisionTreeRegressor(max_depth=3).fit(X,y)
h = float(input("Hours: "))
print("Marks:",m.predict([[h]])[0])

plot_tree(m,filled=True)
plt.show()