from sklearn import linear_model
from numpy import genfromtxt, savetxt

dataset = genfromtxt(open('/Users/larryhan/Dropbox/SML Project2/code/Part B/data_6/365_regression_fetureselected.csv','r'), delimiter=',', dtype='f8')[1:]

target = [x[33] for x in dataset]
train  = [x[0:33] for x in dataset]

clf = linear_model.LinearRegression()
clf.fit(train,target)
print('Variance score: %.2f' % clf.score(train, target))