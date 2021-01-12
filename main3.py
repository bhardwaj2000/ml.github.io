from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()
#print(iris.DESCR)
features = iris.data
labels = iris.target
#print(features[0], labels[0])

clf = KNeighborsClassifier()
clf.fit(features,labels)

pred1 = clf.predict([[5.1, 3.5, 1.4, 0.2]])
print(pred1)

pred2 = clf.predict([[9.1, 31.5, 1.4, 21.2]])
print(pred2)