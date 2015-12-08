README

1 reverse.rb

The training data is in a form that the first user in each line follows all the remaining users in the same line.  This program is to read the original training data file and transform it to another form.  The output data file will be in a form that the first user in each line is followed by all the remaining users in the same line.



2 preprocess.rb

The program reads the original training data, the reverse form of the training data, the subset of the training data and the reverse form of the subset of the training data.
The program samples 1000000 pairs of users in the range of the subset of the training data and generates the features for every pair.
The output of the program is a csv file which includes 1000000 pair of users and its features.



3 testprocess.rb

The program reads the original training data, the reverse form of the training data and the test data as its input.  The program generates features for every pair of users in the test data according to the training data.  The output of the program is a csv file which includes pairs in the test data and its features.

4 random_forest.py

This program take the featured test data and featured training data  as input, output the probability distribution of prediction by random forest.

5 Adaboost_RandomForest

The ensemble algorithm to apply adboost in random forest

6 Logistic_Regression

Code for Logistic Regression

7 CSV_fucntion

Helpers of reading and writing csv


Besides, the Naive Bayes and Decision Tree is simpole algorithm so we directly use Weka to perform the two algorithms. 