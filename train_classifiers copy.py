
from sklearn.model_selection import GridSearchCV, train_test_split
#from sklearn import DecisionTreeClassifier, GaussianNB, SVC, MLPClassifier, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import os
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
import sys

if(len(sys.argv)>1):
    if(sys.argv[1]=='True'):
        verbose = True
    else:
        verbose = False
    if(sys.argv[2]=='True'):
        write_csv=True
    else:
        write_csv=False
        
else:
    
    verbose = False
    write_csv=False
    
path_for_CSV='./RESULTS'

frame = pd.read_csv('./CSV/new_feature_vector_file.csv')

y = frame['buggy'].values
X = frame.drop(columns=['buggy', 'class']).values
weights_bool = y == 0

weights = X[weights_bool]
weights_class_zero = (len(weights)/len(y))
weights_class_one = ((len(y)-len(weights))/len(y))
print(
    f"labels for class 0 training set: {len(weights)}, {weights_class_zero * 100.} %")
print(f"labels for class 1 class training set: {weights_class_one * 100.} %")


if(verbose):
    print("original frame:\n", frame)
    print("Dataset: \n", X)
    print("Labels: \n", y)

training_set_X, test_set_X, training_set_Y, test_set_Y = train_test_split(
    X, y, test_size=0.2, train_size=0.8)
print(
    f"Len test set X: {len(test_set_X)}, Len training set X: {len(training_set_X)}")
print(
    f"Len labels test: {len(test_set_Y)}, Len labels training set: {len(training_set_Y)}")
print()
assert(round(0.8 * len(X)) == len(training_set_X))
weights = {0: weights_class_zero, 1: weights_class_one}
'''ensure that the splitting is correct'''

'''########### STARTING THE TRAINING ############'''

# print(training_set_X)



priors = (None, weights)
model = GaussianNB()
parameters ={'priors': priors, 'var_smoothing': (1e-9, 1e-8, 1e-7)
     }
clf = GridSearchCV(model, parameters, scoring='f1')
clf.fit(training_set_X, training_set_Y)
pred = clf.predict(test_set_X)
print("for Gaussian Model")
print(precision_recall_fscore_support(
                        test_set_Y, pred,  average='binary', zero_division=0))
print(clf.best_params_)

model.fit(training_set_X, training_set_Y)
pred = model.predict(test_set_X)
prec_rec_f1 = precision_recall_fscore_support(
                        test_set_Y, pred,  average='binary', zero_division=0)
prec, rec, f1, _ = prec_rec_f1
print("Gaussian Model")
print(f"precision: {prec}, recall: {rec}, f1: {f1}")


criterion_dec_tree = ('gini', 'entropy')
splitter_dec_tree = ('best', 'random')
list1 = [None]
list2 = range(1, 30)
max_depth_dec_tree = [y for x in [list1, list2] for y in x]
print("max depth: ", max_depth_dec_tree)
min_samples_split_dec_tree = range(2, 30)
model = tree.DecisionTreeClassifier()
parameters ={'criterion': criterion_dec_tree, 'splitter': splitter_dec_tree,\
     'max_depth': max_depth_dec_tree, 'min_samples_split': min_samples_split_dec_tree, \
         'class_weight': (None, weights)}
    
clf = GridSearchCV(model, parameters, scoring='f1')
clf.fit(training_set_X, training_set_Y)
pred = clf.predict(test_set_X)
print(precision_recall_fscore_support(
                        test_set_Y, pred,  average='binary', zero_division=0))
print(clf.best_params_)














