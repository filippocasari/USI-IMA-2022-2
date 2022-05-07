
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


def Dec_Tree(X_train: np.array, X_test, y_train, y_test, weights=None):
    """_summary_

    Args:
        X_train (np.array): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
        weights (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    ''''''
    criterion_dec_tree = ['gini', 'entropy']
    splitter_dec_tree = ['best', 'random']
    list1 = [None]
    list2 = range(1, 30)
    max_depth_dec_tree = [y for x in [list1, list2] for y in x]

    min_samples_split_dec_tree = range(2, 30)
    best_f1_score = 0
    best_params = {'criterion': 0., 'splitting': 0.,'max depth':0.,'min_samples':0.  }
    best_prec_rec_f1 = []
    for crit in criterion_dec_tree:
        for split in splitter_dec_tree:
            for max_dep in max_depth_dec_tree:
                for min_sampl in min_samples_split_dec_tree:

                    model = tree.DecisionTreeClassifier(criterion=crit, splitter=split,
                                                        max_depth=max_dep, min_samples_split=min_sampl, class_weight=weights)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)

                    if(verbose):
                        print(
                            f"with criterion: {crit}, splitting: {split}, max depth: {max_dep}, min_samples: {min_sampl}")
                    prec_rec_f1 = precision_recall_fscore_support(
                        y_test, pred,  average='binary', zero_division=0)
                    if(verbose):
                        print(prec_rec_f1)
                    if(best_f1_score < prec_rec_f1[2]):
                        best_f1_score = prec_rec_f1[2]
                        best_params['criterion'], best_params['max depth'], best_params['min_samples'], best_params['splitting'] = \
                            crit, max_dep, min_sampl, split
                        best_prec_rec_f1 = prec_rec_f1
    return best_f1_score, best_params, best_prec_rec_f1


def SVM(X_train: np.array, X_test: np.array, y_train: np.array, y_test: np.array, weights=None, kernel = 'linear'):
    """ Support Vector Machine (for binary classification)

    Args:
        X_train (np.array): _description_
        X_test (np.array): _description_
        y_train (np.array): _description_
        y_test (np.array): _description_
        weights (_type_, optional): must be a list. Defaults to None.
        kernel (str, optional): kernel, look at sklearn doc. Defaults to 'linear'.

    Returns:
        list: list of object
    """    ''''''
    tolerances=[1e-1, 1e-2]
    max_iterations=[-1]
    C_array=[1.0, 2.0, 3.0, 4.0]
    random_states=range(1, 5)
    best_params = {'tol': 0., 'max_iter': 0.,'C':0.,'random_state':0.  }
    best_prec_rec_f1=[]
    best_f1_score=0.
    for tol in tolerances:
        for max_iter in max_iterations:
            for C in C_array:
                for random_state in random_states:
                    model = SVC(C=C, tol=tol, max_iter=max_iter, class_weight=weights, \
                        random_state=random_state, kernel=kernel)

                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)

                    prec_rec_f1 = precision_recall_fscore_support(
                        y_test, pred,  average='binary', zero_division=0)
                    if(verbose):
                        print(prec_rec_f1)
                    if(best_f1_score < prec_rec_f1[2]):
                        best_f1_score = prec_rec_f1[2]
                        best_params['C'], best_params['max_iter'], best_params['random_state'], best_params['tol'] = \
                            C, max_iter, random_state, tol
                        best_prec_rec_f1 = prec_rec_f1
    return best_f1_score, best_params, best_prec_rec_f1


def write_results(best_params, best_prec_rec_f1, name='' ):
    """ write the results with the best parameters

    Args:
        best_params (array): 
        best_prec_rec_f1 (array): _description_
        name (str, optional): name of the method. Defaults to ''.
    """    ''''''
    print("WITH "+name)
    #crit, split, max_dep, min_sampl = best_params
    
    print("Best Values for this model: ", best_params)
    print(f"precision: {best_prec_rec_f1[0]}, recall: {best_prec_rec_f1[1]},\
         f1: {best_prec_rec_f1[2]}")
    if(write_csv):
        string_to_write= "WITH "+name+"\nBest Values for this model: "+ str(best_params)+f"\nprecision: {best_prec_rec_f1[0]}, recall: {best_prec_rec_f1[1]},\
            f1: {best_prec_rec_f1[2]}\n"
        if(os.path.isdir(path_for_CSV) == False):
                os.mkdir(path_for_CSV)
        f = open(path_for_CSV+'/results.txt', "a")
        f.write(string_to_write)
        f.close()
        
    
    
def NeuralNetwork(X_train, X_test, y_train, y_test):
    activations=['identity', 'logistic', 'tanh', 'relu']
    solvers=['lbfgs', 'sgd', 'adam']
    learning_rates=['constant', 'invscaling', 'adaptive']
    shuffles=[True, False]
    best_params = {'activation': 0., 'solver': 0.,'shuffle':True,'learning_rate':0.  }
    best_prec_rec_f1=[]
    best_f1_score=0.

    for shuffle in shuffles:
        for solver in solvers:
            for lr in learning_rates:
                for act in activations:
                    model = MLPClassifier(activation=act, solver=solver, shuffle=shuffle, learning_rate=lr)
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    prec_rec_f1 = precision_recall_fscore_support(
                        y_test, pred,  average='binary', zero_division=0)
                    if(verbose):
                        print(prec_rec_f1)
                    if(best_f1_score < prec_rec_f1[2]):
                        best_f1_score = prec_rec_f1[2]
                        
                        best_prec_rec_f1 = prec_rec_f1
                        best_params['activation'], best_params['learning_rate'], best_params['shuffle'], best_params['solver']= \
                            act, lr, shuffle, solver
    return best_f1_score, best_params, best_prec_rec_f1


def RandomForest(X_train: np.array, X_test, y_train, y_test, weights=None):
    """Random Forest, tested several parameters

    Args:
        X_train (_type_): _description_
        X_test (_type_): _description_
        y_train (_type_): _description_
        y_test (_type_): _description_
        weights (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    ''''''
    
    n_estimators_array=[100, 150, 200, 300]
    criterions=['gini', 'entropy']
    min_samples_splits=[2, 3, 4]
    max_features_array=['auto', 'sqrt', 'log2']
    bootstraps=[True, False]
    best_params = {'n_estimators': 0., 'criterion': 0.,'max_features':'auto','bootstrap':True, 'min_samples_splits':2  }
    best_prec_rec_f1=[]
    best_f1_score=0.
    for n_estinamtors in n_estimators_array:
        for criterion in criterions:
            for min_samples_split in min_samples_splits:
                for max_features in max_features_array:
                    for bootstrap in bootstraps:
                        model = RandomForestClassifier(n_estimators=n_estinamtors, criterion=criterion, \
                            min_samples_split=min_samples_split,\
                                 bootstrap=bootstrap, max_features=max_features, class_weight=weights)
                        model.fit(X=X_train, y=y_train)
                        pred= model.predict(X_test)
                        prec_rec_f1 = precision_recall_fscore_support(
                        y_test, pred,  average='binary', zero_division=0)
                        
                        if(best_f1_score < prec_rec_f1[2]):
                            best_f1_score = prec_rec_f1[2]
                            best_params['bootstrap'], best_params['criterion'], best_params['max_features'], best_params['min_samples_splits'], best_params['n_estimators'] = \
                                bootstrap, criterion, max_features, min_samples_split, n_estinamtors
                            best_prec_rec_f1 = prec_rec_f1
    return best_f1_score, best_params, best_prec_rec_f1


best_f1_score, best_params, best_prec_rec_f1 = \
    Dec_Tree(training_set_X, test_set_X, y_test=test_set_Y,
             y_train=training_set_Y, weights=None)
write_results(best_params, best_prec_rec_f1, 'Decision Tree unbalanced')



best_f1_score, best_params, best_prec_rec_f1 = \
    Dec_Tree(training_set_X, test_set_X, y_test=test_set_Y,
             y_train=training_set_Y, weights=weights)
write_results(best_params, best_prec_rec_f1, 'Decision Tree balanced')

'''crit, split, max_dep, min_sampl = best_params
print(
    f"with criterion: {crit}, splitting: {split}, max depth: {max_dep}, min_samples: {min_sampl} we have the best f1 score")
print(f"precision: {best_prec_rec_f1[0]}, recall: {best_prec_rec_f1[1]}, f1: {best_prec_rec_f1[2]}")'''



model = GaussianNB()
model.fit(training_set_X, training_set_Y)
pred = model.predict(test_set_X)
prec_rec_f1 = precision_recall_fscore_support(
                        test_set_Y, pred,  average='binary', zero_division=0)
prec, rec, f1, _ = prec_rec_f1
print("Gaussian Model")
print(f"precision: {prec}, recall: {rec}, f1: {f1}")


best_f1_score, best_params, best_prec_rec_f1 = \
    SVM(X_train=training_set_X, X_test=test_set_X, y_test=test_set_Y,
             y_train=training_set_Y)
write_results(best_params, best_prec_rec_f1, 'SVC unbalanced')
'''
best_f1_score, best_params, best_prec_rec_f1 = \
    SVM(X_train=training_set_X, X_test=test_set_X, y_test=test_set_Y,
             y_train=training_set_Y, weights=weights)

write_results(best_params, best_prec_rec_f1, 'SVC balanced')'''


'''
criterion_dec_tree = ('gini', 'entropy')
splitter_dec_tree = ('best', 'random')
list1 = [None]
list2 = range(1, 30)
max_depth_dec_tree = [y for x in [list1, list2] for y in x]

min_samples_split_dec_tree = range(2, 30)
model = tree.DecisionTreeClassifier()
parameters ={'criterion': criterion_dec_tree, 'splitter': splitter_dec_tree,\
     'max_depth': max_depth_dec_tree, 'min_samples_split': min_samples_split_dec_tree}
    
clf = GridSearchCV(model, parameters)
clf.fit(training_set_X, training_set_Y)
pred = clf.predict(test_set_X)
print(precision_recall_fscore_support(
                        test_set_Y, pred,  average='binary', zero_division=0))
print(clf.best_params_)'''




best_f1_score, best_params, best_prec_rec_f1 = \
    NeuralNetwork(X_train=training_set_X, X_test=test_set_X, y_test=test_set_Y,
             y_train=training_set_Y)
write_results(best_params, best_prec_rec_f1, 'MLPClassifier')



start=time.time()
best_f1_score, best_params, best_prec_rec_f1 = \
    RandomForest(X_train=training_set_X, X_test=test_set_X, y_test=test_set_Y,
             y_train=training_set_Y)
write_results(best_params, best_prec_rec_f1, 'Random Forest unbalanced')
end=time.time()
print("time execution: ", end-start)
start=time.time()
best_f1_score, best_params, best_prec_rec_f1 = \
    RandomForest(X_train=training_set_X, X_test=test_set_X, y_test=test_set_Y,
             y_train=training_set_Y, weights=weights)
write_results(best_params, best_prec_rec_f1, 'Random Forest balanced')
end=time.time()
print("time execution: ", end-start)






