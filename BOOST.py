import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

X = [[1],[2],[3],[4],[5],[6]]
y = [0,0,0,1,1,1]

m = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),n_estimators=5).fit(X,y)
p = m.predict(X)
print("Accuracy:",accuracy_score(y,p))

ConfusionMatrixDisplay.from_predictions(y,p)
plt.show()
