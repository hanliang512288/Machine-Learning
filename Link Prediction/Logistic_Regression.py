__author__ = 'sym44'

from sklearn.linear_model import LogisticRegression
import csv_function
from sklearn import metrics

def main():
    # read in the training data and test data
    trainset = csv_function.read_csv("train_14.csv")
    testset = csv_function.read_csv("test_14.csv")

    # the first column of the training set will be the target for the classifier
    target = [x[17] for x in trainset]
    train = [x[3:16] for x in trainset]
    test = [x[3:16] for x in testset]

    # create and train the logistic regression
    model = LogisticRegression()
    model.fit(train, target)
    print(model)

    # make predictions
    expected = target
    predicted = model.predict(train)

    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

    # make predictions
    pred_test = model.predict(test)
    csv_function.write_csv("submission_lr.csv", list(pred_test))

if __name__=="__main__":
    main()