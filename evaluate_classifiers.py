
from joblib import dump, load
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
import pandas as pd
from sklearn.model_selection import KFold, RepeatedKFold
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
from scipy.stats import wilcoxon
import warnings
warnings.filterwarnings("ignore")
path_models = './MODELS/'
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
def Cross_Validation_Scores(clf, X, y):
    """Cross Validation function. returns scores (f1 score, precision, recall)

    Args:
        clf (_type_): model that must be validated
        X (_type_): X: dataset
        y (_type_): y: labels

    Returns:
        _type_: f1 score, precision, recall
    """    ''''''

    f1_scores = []
    prec_scores = []
    rec_scores = []
    k_fold_stratified = StratifiedKFold(shuffle=True)
    for i in range(20):
        scores = cross_validate(clf, X, y,
                                scoring={'f1': 'f1', 'precision': 'precision',
                                         'recall': 'recall'}, n_jobs=-1, cv=k_fold_stratified)
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

    string_ = f"For model {model}-> f1 score :{f1.mean()}, prec: {prec.mean()}, rec: {rec.mean()}\n\n"
    print(string_)
    return string_


frame = pd.read_csv('./CSV/new_feature_vector_file.csv')
'''read the dataset'''

y = frame['buggy'].values
X = frame.drop(columns=['buggy', 'class']).values



std_scaler = StandardScaler()
X_nn = std_scaler.fit_transform(X)
del std_scaler

weights_bool = y == 0

weights = X[weights_bool]
weights_class_zero = (len(weights)/len(y))
weights_class_one = ((len(y)-len(weights))/len(y))
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8)
X_train_nn, X_test_nn, Y_train_nn, Y_test_nn = train_test_split(
    X, y, test_size=0.2, train_size=0.8)


bias_classifier = np.zeros(X_test.shape[0])
print(f"shape of bias classifier: {bias_classifier.shape}")
print(f"shape of test set : {Y_test.shape}")


Y_test = Y_test.astype(np.float32)

f1_bias_classifier = np.array(precision_recall_fscore_support(Y_test, bias_classifier))[:-1]
#prec_bias_classifier = precision_score(Y_test, bias_classifier)
#rec_bias_classifier = recall_score(Y_test, bias_classifier)
print(f"shape f1 bias cf : {f1_bias_classifier.shape}")
print(f1_bias_classifier.mean(axis=1))
print(Y_test)
print(bias_classifier)


f1_bias =[]
prec_bias =[]
rec_bias =[]
k_fold_stratified = StratifiedKFold(shuffle=True)

print(k_fold_stratified)
for i in range(20):
    k_fold_stratified = StratifiedKFold(shuffle=True)
    for train_index, test_index in k_fold_stratified.split(X, y):
        X_test_bias, X_test_bias = X[train_index], X[test_index]
        y_test_bias, y_test_bias = y[train_index], y[test_index]
        prec,rec,f1 = np.array(precision_recall_fscore_support(Y_test, bias_classifier))[:-1].mean(axis=1)
        f1_bias.append(f1)
        rec_bias.append(rec)
        prec_bias.append(prec)
        
        
        

#time.sleep(100)

'''
while(np.all(Y_test == 0)):
    
    X_train, X_test ,Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, train_size=0.8)'''

#weights = {0: weights_class_zero, 1: weights_class_one}
# print(weights)

path_step_4 = 'step_4_results.txt'


f = open(path_step_4, "w")


clf = load(path_models+'DecisionTree.joblib')
f1_scores_dec_tree, prec_scores_dec_tree, rec_scores_dec_tree = Cross_Validation_Scores(
    clf, X=X, y=y)
string = print_results(f1_scores_dec_tree, prec_scores_dec_tree,
                       rec_scores_dec_tree, model='Decision Tree')
f.write(string)

clf = load(path_models+'GaussianNB.joblib')
f1_scores_gauss, prec_scores_gauss, rec_scores_gauss = Cross_Validation_Scores(
    clf, X=X, y=y)
string = print_results(f1_scores_gauss, prec_scores_gauss,
                       rec_scores_gauss, model='Gaussian NB')
f.write(string)

clf = load(path_models+'LinearSVC.joblib')
f1_scores_svc, prec_scores_svc, rec_scores_svc = Cross_Validation_Scores(
    clf, X=X, y=y)
string = print_results(f1_scores_svc, prec_scores_svc,
                       rec_scores_svc, model='SVC')
f.write(string)

clf = load(path_models+'MPLClassifier.joblib')
f1_scores_mpl, prec_scores_mpl, rec_scores_mpl = Cross_Validation_Scores(
    clf, X=X_nn, y=y)
string = print_results(f1_scores_mpl, prec_scores_mpl,
                       rec_scores_mpl, model='MPL')
f.write(string)

clf = load(path_models+'RandomForestClassifier.joblib')
f1_scores_rf, prec_scores_rf, rec_scores_rf = Cross_Validation_Scores(
    clf, X=X, y=y)
string = print_results(f1_scores_rf, prec_scores_rf,
                       rec_scores_rf, model='Random Forest')
f.write(string)


f.close()

f1 = [f1_scores_rf, f1_scores_mpl, f1_scores_svc,
      f1_scores_gauss, f1_scores_dec_tree]
precisions = f1 = [prec_scores_rf, prec_scores_mpl, prec_scores_svc,
      prec_scores_gauss, prec_scores_dec_tree]
recalls = [rec_scores_rf, rec_scores_mpl, rec_scores_svc,
      rec_scores_gauss, rec_scores_dec_tree]
models = {'Random Forest': f1[0], 'MPL': f1[1],
          'SVC': f1[2], 'GaussNB': f1[3], 'Decision Tree': f1[4], 'Bias Classifier': f1_bias}
models_prec = {'Random Forest': precisions[0], 'MPL': precisions[1],
          'SVC': precisions[2], 'GaussNB': precisions[3], 'Decision Tree': precisions[4], 'Bias Classifier': prec_bias}

models_rec = {'Random Forest': recalls[0], 'MPL': recalls[1],
          'SVC': recalls[2], 'GaussNB': recalls[3], 'Decision Tree': recalls[4], 'Bias Classifier': rec_bias}


models_names = ['Random Forest', 'MPL', 'SVC', 'GaussNB', 'Decision Tree', 'Bias Classifier']
for name1 in models_names:
    for name2 in models_names:
        if(name2 != name1):
            w, p = wilcoxon(models[name1], models[name2])
            print(f"f1 couple [ {name1}, {name2} ], p value: {p}")
            w, p = wilcoxon(models_prec[name1], models_prec[name2])
            print(f" precision couple [ {name1}, {name2} ], p value: {p}")
            w, p = wilcoxon(models_rec[name1], models_rec[name2])
            print(f"recall couple [ {name1}, {name2} ], p value: {p}")
            print("\n ******************* \n")


models_f1 = pd.DataFrame(models)
models_prec = pd.DataFrame(models_prec)
models_rec = pd.DataFrame(models_rec)
models_f1.to_csv('f1_dataframe.csv')
models_prec.to_csv('prec_dataframe.csv')
models_rec.to_csv('rec_dataframe.csv')



    








