import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]
y = [20,30,46,50,65,70,80,90,95,100]

m = LinearRegression().fit(X,y)
print(m.predict([[7.5]]))

plt.scatter(X,y)
plt.plot(X,m.predict(X),'r')
plt.show()