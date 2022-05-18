
from joblib import dump, load
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np
from sklearn.preprocessing import StandardScaler

from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings("ignore")
path_models= './MODELS/'


def Cross_Validation_Scores(clf, X, y):
    """Cross Validation function. returns scores (f1 score, precision, recall)

    Args:
        clf (_type_): model that must be validated
        X (_type_): X: dataset
        y (_type_): y: labels

    Returns:
        _type_: f1 score, precision, recall
    """    ''''''
    f1_scores =[]
    prec_scores =[]
    rec_scores =[]
    k_fold_stratified = StratifiedKFold(shuffle=True)
    for i in range(20):
        scores = cross_validate(clf, X, y,\
                                scoring={'f1': 'f1', 'precision': 'precision',
        'recall': 'recall'}, n_jobs=-1 ,cv = k_fold_stratified )
        f1_scores.append(scores['test_f1'])
        prec_scores.append(scores['test_precision'])
        rec_scores.append(scores['test_recall'])
        
    f1_scores = np.array(f1_scores).flatten()
    prec_scores = np.array(prec_scores).flatten()
    rec_scores = np.array(rec_scores).flatten()
    return f1_scores, prec_scores, rec_scores


def print_results(f1, prec, rec, model):
    """print the results

    Args:
        f1 (array of float): f1 score
        prec (array of float): precision
        rec (array of float): recall
        model (str): model name

    Returns:
        str: string to print (or write into a file eventually)
    """    ''''''
    
    string_=f"For model {model}-> f1 score :{f1.mean()}, prec: {prec.mean()}, rec: {rec.mean()}\n\n"
    print(string_)
    return string_

frame = pd.read_csv('./CSV/new_feature_vector_file.csv')

y = frame['buggy'].values
X = frame.drop(columns=['buggy', 'class']).values

std_scaler = StandardScaler()
X_nn = std_scaler.fit_transform(X)
del std_scaler

weights_bool = y == 0

weights = X[weights_bool]
weights_class_zero = (len(weights)/len(y))
weights_class_one = ((len(y)-len(weights))/len(y))
X_train, X_test ,Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, train_size=0.8)
X_train_nn, X_test_nn ,Y_train_nn, Y_test_nn = train_test_split(
        X, y, test_size=0.2, train_size=0.8)

'''
while(np.all(Y_test == 0)):
    
    X_train, X_test ,Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, train_size=0.8)'''
        
#weights = {0: weights_class_zero, 1: weights_class_one}
#print(weights)

path_step_4 = '.step_4_results.txt'


f = open(path_step_4, "w")


clf = load(path_models+'DecisionTree.joblib')
f1_scores_dec_tree,prec_scores_dec_tree,rec_scores_dec_tree =Cross_Validation_Scores(clf, X=X, y=y)
string = print_results(f1_scores_dec_tree,prec_scores_dec_tree,rec_scores_dec_tree, model='Decision Tree')
f.write(string)

clf = load(path_models+'GaussianNB.joblib')
f1_scores_gauss,prec_scores_gauss,rec_scores_gauss =Cross_Validation_Scores(clf, X=X, y=y)
string=print_results(f1_scores_gauss,prec_scores_gauss,rec_scores_gauss, model='Gaussian NB')
f.write(string)

clf = load(path_models+'LinearSVC.joblib')
f1_scores_svc,prec_scores_svc,rec_scores_svc =Cross_Validation_Scores(clf, X=X, y=y)
string=print_results(f1_scores_svc,prec_scores_svc,rec_scores_svc, model='SVC')
f.write(string)

clf = load(path_models+'MPLClassifier.joblib')
f1_scores_mpl,prec_scores_mpl,rec_scores_mpl =Cross_Validation_Scores(clf, X=X_nn, y=y)
string=print_results(f1_scores_mpl,prec_scores_mpl,rec_scores_mpl, model='MPL')
f.write(string)

clf = load(path_models+'RandomForestClassifier.joblib')
f1_scores_rf,prec_scores_rf,rec_scores_rf =Cross_Validation_Scores(clf, X=X, y=y)
string=print_results(f1_scores_rf,prec_scores_rf,rec_scores_rf, model='Random Forest')
f.write(string)

f.close()

print(X.shape)
print(y.shape)
'''TODO: still working on it'''
w, p = wilcoxon(X, np.zeros(y.shape))
print(f"w: {w} , p: {p}")










