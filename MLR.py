import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[2,65,50],[3,70,55],[4,72,60],[5,75,65],[6,80,70],[7,82,75],[8,85,80],[9,90,85],[10,92,88]])
y = [48,55,61,67,74,79,85,91,95]

m = LinearRegression().fit(X,y)
h,a,s = float(input("Hours: ")),float(input("Attendance: ")),float(input("Assignment: "))
p = m.predict([[h,a,s]])
print("Marks:",p[0])

plt.scatter(m.predict(X),y)
plt.plot(y,y,'r')
plt.show()
