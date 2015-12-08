import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold
from numpy import genfromtxt, savetxt

from sklearn import tree
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification
from numpy import genfromtxt, savetxt

dataset = genfromtxt(open('/Users/larryhan/Dropbox/SML Project2/code/Part B/data_4/classification.csv','r'), delimiter=',', dtype='f8')[1:]
title =  genfromtxt(open('/Users/larryhan/Dropbox/SML Project2/code/Part B/data_4/classification.csv','r'), delimiter=',', dtype='S5')[0]

target = [x[418] for x in dataset]

train  = [x[0:418] for x in dataset]


sel = VarianceThreshold(0.9*(1-0.9))
selected = sel.fit_transform(train)

support = sel.get_support()

sub_title = []
for i in range(len(support)):
	if support[i]:
		sub_title.append(title[i])

# print(sub_title)


svc = svm.LinearSVC()

# clf = tree.DecisionTreeClassifier()

rfecv = RFECV(svc, step=1, cv=StratifiedKFold(target, 2), scoring='accuracy')
rfecv.fit(selected,target)

rf_support = rfecv.support_

# print(rf_support)

sub2_title = []
for i in range(len(rf_support)):
	if (rf_support[i]):
		sub2_title.append(sub_title[i])
print sub2_title
# 
# print(array)
print(rfecv.score(selected,target))


print("Optimal number of features : %d" % rfecv.n_features_)
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()