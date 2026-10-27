import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

X = np.arange(1,11).reshape(-1,1)
y = [0,0,0,0,0,1,1,1,1,1]

m = LogisticRegression().fit(X,y)
h = float(input("Hours: "))
p = m.predict_proba([[h]])[0,1]
print("PASS" if p>=0.5 else "FAIL")

plt.scatter(X,y)
plt.plot(X,m.predict_proba(X)[:,1],'r')
plt.show()
