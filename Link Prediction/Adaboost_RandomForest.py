__author__ = 'sym44'

import csv_function
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

def main():
    # read in the training data and test data
    trainset = csv_function.read_csv("train_14.csv")
    testset = csv_function.read_csv("test_14.csv")

    # the first column of the training set will be the target for the classifier
    target = [x[17] for x in trainset]
    train = [x[3:16] for x in trainset]
    test = [x[3:16] for x in testset]

    # create and fit the adaboost model
    model = AdaBoostClassifier(DecisionTreeClassifier(), algorithm="SAMME", n_estimators=200)
    model.fit(train, target)

    # predicted = model.predict(train)
    # expected = target
    #
    # # summarize the fit of the model
    # print(metrics.classification_report(expected, predicted))
    # print(metrics.confusion_matrix(expected, predicted))

    predict_test = model.predict(test)
    csv_function.write_csv("submission_ada.csv", list(predict_test))

if __name__=="__main__":
    main()