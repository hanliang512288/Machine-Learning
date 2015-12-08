from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt


#create the training & test sets, skipping the header row with [1:]
dataset = genfromtxt(open('train_100W_14f.csv','r'), delimiter=',', dtype='f8')[1:]

target = [x[17] for x in dataset]
train = [x[3:17] for x in dataset]

print('Training data input finished')

test = genfromtxt(open('test_14f.csv','r'), delimiter=',', dtype='f8')[1:]
print(len(test))
print('Test data input finished')
#create and train the random forest
#multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
rf = RandomForestClassifier(n_estimators=200,n_jobs=-1)

print('Building finished')
rf.fit(train, target)

savetxt('random_forest_100W_14f_200tr-2.csv', rf.predict_proba(test), delimiter=',', fmt='%f')