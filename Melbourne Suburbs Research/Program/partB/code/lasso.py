from sklearn import linear_model
from numpy import genfromtxt, savetxt
from sklearn.feature_selection import VarianceThreshold


from sklearn.feature_selection import VarianceThreshold

dataset = genfromtxt(open('/Users/larryhan/Dropbox/SML Project2/code/Part B/data_5/365_regression.csv','r'), delimiter=',', dtype='f8')[1:]
title =  genfromtxt(open('/Users/larryhan/Dropbox/SML Project2/code/Part B/data_5/365_regression.csv','r'), delimiter=',', dtype='S5')[0]

target = [x[716] for x in dataset]
train  = [x[0:716] for x in dataset]

print (len(train[0]))

# sel = VarianceThreshold(0.95*(1-0.95))
# selected = sel.fit_transform(train)

# support = sel.get_support()

# sub_title = []
# for i in range(len(support)):
# 	if support[i]:
# 		sub_title.append(title[i])

# print(len(selected[0]))

selected = train
sub_title = title


model = linear_model.Lasso(alpha=10)
model.fit(selected,target)
score = model.score(selected,target)

print(model.coef_)
print(score)
print(model.intercept_)

sub2_title = []
for i in range(len(model.coef_)):
	if (model.coef_[i] != 0):
		sub2_title.append(sub_title[i])
print sub2_title
print len(sub2_title)
