import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

X = [[1],[2],[3],[4],[5],[6]]
y = [0,0,0,1,1,1]

m = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=3).fit(X,y)
p = m.predict(X)
print("Accuracy:",accuracy_score(y,p))

ConfusionMatrixDisplay.from_predictions(y,p)
plt.show()
