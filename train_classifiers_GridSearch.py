
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import os
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
import sys
import os.path
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

'''NOTES: for running the file: python3 "name_of_this_file" "args" '''


'''This is an other way to Training and finding the best hyperparamters\
    It uses the sklearn class so-called GridSearch'''

verbose = False
write_csv = False
save_models = False

if(len(sys.argv) > 1 and len(sys.argv) < 5):
    '''if any args is given'''

    if(sys.argv[1] == 'True'):
        '''set verbose '''

        verbose = True

    if(sys.argv[2] == 'True'):
        '''set if this program has to write a csv file'''
        write_csv = True

    if(sys.argv[3] == 'True'):
        '''saving models'''
        save_models = True


path_for_CSV = './RESULTS'
path_models = './MODELS'
if(os.path.isdir(path_models) == False):
    os.mkdir(path_models)

frame = pd.read_csv('./CSV/new_feature_vector_file.csv').drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
print(frame)
y = frame['buggy'].values
X = frame.drop(columns=['buggy', 'class']).values
weights_bool = y == 0
weights = X[weights_bool]
weights_class_zero = (len(weights)/len(y))
weights_class_one = ((len(y)-len(weights))/len(y))

std_scaler = StandardScaler()
std_scaler.fit(X=X)

X_nn = std_scaler.transform(X)
del std_scaler


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
    X, y, test_size=0.2, train_size=0.8, shuffle=True)

training_set_X_nn, test_set_X_nn, training_set_Y_nn, test_set_Y_nn = train_test_split(
    X_nn, y, test_size=0.2, train_size=0.8, shuffle=True)
print(
    f"Len test set X: {len(test_set_X)}, Len training set X: {len(training_set_X)}")
print(
    f"Len labels test: {len(test_set_Y)}, Len labels training set: {len(training_set_Y)}")
print()
assert(int(0.8 * len(X)) == len(training_set_X))
weights = {0: weights_class_zero, 1: weights_class_one}
'''ensure that the splitting is correct'''

'''########### STARTING THE TRAINING ############'''

# print(training_set_X)


def GridSearchFun(model, metric, parameters, X_train, X_test, Y_train, Y_test, model_name='', weights=None):
    """Grid Search to find best hyperparameters

    Args:
        model (any): model 
        metric (string): choose the best model according with the metric
        parameters (_type_): set of different parameters
        X_train (_type_): training set (X)
        X_test (_type_): test set (X)
        Y_train (_type_): training set (Y)
        Y_test (_type_): test set (Y)
        model_name (str, optional): model's name. Defaults to ''.
    """    ''''''
    print(f"training model: {model_name}")
    clf = GridSearchCV(model, parameters, scoring=[
                       'f1', 'recall', 'precision'], n_jobs=-1,  refit='f1', cv=StratifiedKFold())
    clf.fit(X_train, Y_train)
    pred = clf.predict(X_test)
    prec, recall, f1beta, _ = precision_recall_fscore_support(
        Y_test, pred,  average='binary', zero_division=0)
    f1 = 2.*(prec*recall)/(prec+recall)
    print("precision, recall, f1 = ",)
    print(f"Best parameters for {model_name} are: \n {clf.best_params_}")
    if(write_csv):
        string_to_write = f"Best parameters for {model_name} are : \n{clf.best_params_} \n f1= {f1}, \
            precision: {prec}, recall= {recall}\n ****************************\n "

        if(os.path.isdir(path_for_CSV) == False):
            os.mkdir(path_for_CSV)

        f = open(path_for_CSV+'/results_grid_search.txt', "a")
        f.write(string_to_write)
        f.close()
    return clf.best_estimator_, pd.DataFrame(clf.cv_results_).round(4)


print(set(test_set_Y) - set(training_set_Y))

priors = (None, weights)
model = GaussianNB()
parameters = {'priors': priors, 'var_smoothing': (1e-9, 1e-8, 1e-7)
              }
best_estimator, table_results = GridSearchFun(model, 'balanced_accuracy', parameters, training_set_X, test_set_X,
                                              training_set_Y, test_set_Y, 'Gaussian Naive Bayes')
if(save_models):
    dump(best_estimator, path_models+'/GaussianNB.joblib')
filter_col = [col for col in table_results if col.startswith('params') or (col.endswith('f1') and col.startswith('mean')) or \
    (col.endswith('precision') and col.startswith('mean')) or (col.endswith('recall') and col.startswith('mean'))]
table_results = table_results[filter_col].fillna(0)
print(table_results.head())
table_results.replace(0.0, 0.0000)
table_results.to_csv("./CSV/GaussianNB_metrics.csv")


criterion_dec_tree = ('gini', 'entropy')
splitter_dec_tree = ('best', 'random')
list1 = [None]
list2 = range(1, 30)
max_depth_dec_tree = [y for x in [list1, list2] for y in x]
min_samples_split_dec_tree = range(2, 30)
model = tree.DecisionTreeClassifier()

parameters = {'criterion': criterion_dec_tree, 'splitter': splitter_dec_tree,
              'max_depth': max_depth_dec_tree, 'min_samples_split': min_samples_split_dec_tree,
              'class_weight': (None, weights)}
best_estimator, table_results = GridSearchFun(model, 'balanced_accuracy', parameters, training_set_X, test_set_X,
                                              training_set_Y, test_set_Y, 'Decision Tree ')
if(save_models):
    dump(best_estimator, path_models+'/DecisionTree.joblib')
filter_col = [col for col in table_results if col.startswith('params') or (col.endswith('f1') and col.startswith('mean')) or \
    (col.endswith('precision') and col.startswith('mean')) or (col.endswith('recall') and col.startswith('mean'))]
table_results = table_results[filter_col].fillna(0)
print(table_results.head())
table_results.replace(0.0, 0.0000)
table_results.to_csv("./CSV/DecisionTree_metrics.csv")

max_iters = (200,  400, 700)
activations = ('identity', 'logistic', 'tanh', 'relu')
solvers = ('lbfgs', 'sgd', 'adam')
learning_rates = ('constant', 'invscaling', 'adaptive')
shuffles = (True, False)
parameters = {'activation': activations, 'solver': solvers,
              'learning_rate': learning_rates, 'shuffle': shuffles, 'max_iter': max_iters}
model = MLPClassifier()
best_estimator, table_results = GridSearchFun(model, 'balanced_accuracy', parameters,
                                              training_set_X_nn, test_set_X_nn, training_set_Y_nn, test_set_Y_nn, 'Neural Networks')

if(save_models):
    dump(best_estimator, path_models+'/MPLClassifier.joblib')
filter_col = [col for col in table_results if col.startswith('params') or (col.endswith('f1') and col.startswith('mean')) or \
    (col.endswith('precision') and col.startswith('mean')) or (col.endswith('recall') and col.startswith('mean'))]
table_results = table_results[filter_col].fillna(0)
table_results.replace(0.0, 0.0000)
print(table_results.head())
table_results.to_csv("./CSV/MPLClassifier_metrics.csv")
n_estimators_array = (100, 150, 200, 300)
criterions = ('gini', 'entropy')
min_samples_splits = (2, 3, 4)
max_features_array = ('auto', 'sqrt', 'log2')
bootstraps = (True, False)
model = RandomForestClassifier()
parameters = {'n_estimators': n_estimators_array, 'criterion': criterions,
              'min_samples_split': min_samples_splits, 'max_features': max_features_array, 'bootstrap': bootstraps}
best_estimator, table_results = GridSearchFun(model, 'balanced_accuracy', parameters, training_set_X, test_set_X,
                                              training_set_Y, test_set_Y, 'Random Forest')

if(save_models):
    dump(best_estimator, path_models+'/RandomForestClassifier.joblib')
filter_col = [col for col in table_results if col.startswith('params') or (col.endswith('f1') and col.startswith('mean')) or \
    (col.endswith('precision') and col.startswith('mean')) or (col.endswith('recall') and col.startswith('mean'))]
table_results = table_results[filter_col].fillna(0)
table_results.replace(0.0, 0.0000)
print(table_results.head())
table_results.to_csv("./CSV/RF_metrics.csv")
#C_array=(1., 2., 3., 4.)
penalties = ('l1', 'l2')
losses = ('hinge', 'squared_hinge')
multi_classes = ('ovr', 'crammer_singer')

model = LinearSVC()
parameters = {'penalty': penalties,
              'loss': losses, 'multi_class': multi_classes}


best_estimator, table_results = GridSearchFun(model, 'balanced_accuracy', parameters, training_set_X, test_set_X,
                                              training_set_Y, test_set_Y, 'Support Vector Machine (linear)')
if(save_models):
    dump(best_estimator, path_models+'/LinearSVC.joblib')
filter_col = [col for col in table_results if col.startswith('params') or (col.endswith('f1') and col.startswith('mean')) or \
    (col.endswith('precision') and col.startswith('mean')) or (col.endswith('recall') and col.startswith('mean'))]
table_results = table_results[filter_col].fillna(0)
print(table_results.head())
table_results.replace(0.0, 0.0000)
table_results.to_csv("./CSV/SVC_metrics.csv")
