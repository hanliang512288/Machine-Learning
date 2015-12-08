from sklearn import svm
from numpy import genfromtxt, savetxt
from sklearn import cross_validation
from sklearn.feature_selection import VarianceThreshold

dataset = genfromtxt(open('/Users/larryhan/Dropbox/SML Project2/code/Part B/data_4/classification_11f.csv','r'), delimiter=',', dtype='f8')[1:]
target = [x[11] for x in dataset]
train  = [x[0:11] for x in dataset]

dataset_test = genfromtxt(open('/Users/larryhan/Dropbox/SML Project2/code/Part B/data_5/365_classification_11f.csv','r'), delimiter=',', dtype='f8')[1:]
target_test = [x[11] for x in dataset_test]
train_test = [x[0:11] for x in dataset_test]
# sel = VarianceThreshold(0.9*(1-0.9))
# selected = sel.fit_transform(train)

clf = svm.LinearSVC()

# scores = cross_validation.cross_val_score(clf,train,target, cv=5)
clf.fit(train,target)
print(clf.score(train_test,target_test))
# print(scores.mean())