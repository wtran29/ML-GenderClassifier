from sklearn import tree, neighbors, svm, metrics, linear_model
import numpy as np


# [height, width, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37],
     [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

# Classifiers
clf = tree.DecisionTreeClassifier()
clf_svc = svm.SVC()
clf_pt = linear_model.Perceptron()
neigh = neighbors.KNeighborsClassifier()

# Training models
clf = clf.fit(X, Y)
clf_svc = clf_svc.fit(X, Y)
clf_pt = clf_pt.fit(X, Y)
neigh = neigh.fit(X, Y)

# Testing the same data
predict_clf = clf.predict(X)
acc_dtc = metrics.accuracy_score(Y, predict_clf) * 100
result_dtc = clf.predict([[190, 70, 43]])
print('Accuracy for DecisionTreeClassifier: {}'.format(acc_dtc))
print(result_dtc)

predict_svc = clf_svc.predict(X)
acc_svc = metrics.accuracy_score(Y, predict_svc) * 100
result_svc = clf_svc.predict([[190, 70, 43]])
print('Accuracy for SectorVectorClass: {}'.format(acc_svc))
print(result_svc)

predict_pt = clf_pt.predict(X)
acc_pt = metrics.accuracy_score(Y, predict_pt) * 100
result_pt = clf_pt.predict([[190, 70, 43]])
print('Accuracy for Perceptron: {}'.format(acc_pt))
print(result_pt)

predict_knn = neigh.predict(X)
acc_knn = metrics.accuracy_score(Y, predict_knn) * 100
result_knn = neigh.predict([[190, 70, 43]])
print('Accuracy for KNearestNeighbor: {}'.format(acc_knn))
print(result_knn)

# Determine which classifier is best
index = np.argmax([acc_svc, acc_pt, acc_knn])
classifiers = {0: 'DTC', 1: 'SVC', 2: 'Perceptron', 3: 'KNN'}
print('Best Gender Classifier is {}'.format(classifiers[index]))




