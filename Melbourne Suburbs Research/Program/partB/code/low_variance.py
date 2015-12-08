from sklearn.feature_selection import VarianceThreshold
from numpy import genfromtxt, savetxt

dataset = genfromtxt(open('/Users/larryhan/Dropbox/SML Project2/code/Part B/data_4/classification.csv','r'), delimiter=',', dtype='f3')[1:]
title = genfromtxt(open('/Users/larryhan/Dropbox/SML Project2/code/Part B/data_4/classification.csv','r'), delimiter=',',dtype="S5")[0]

target = [x[418] for x in dataset]
train  = [x[0:418] for x in dataset]

sel = VarianceThreshold(0.9*(1-0.9))
sel.fit_transform(train)


support = sel.get_support()

# for i in range(len(support)):
# 	if support[i]:
# 		print(title[i])


sub_title = []
for i in range(len(support)):
	if support[i]:
		sub_title.append(title[i])

