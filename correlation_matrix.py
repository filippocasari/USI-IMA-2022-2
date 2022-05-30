from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
frame = pd.read_csv('./CSV/new_feature_vector_file.csv').drop(columns=[ 'class','Unnamed: 0.1', 'Unnamed: 0'])
'''read the dataset'''
frame_clean = frame

y = frame['buggy']
X = frame_clean.drop(columns=['buggy'])
X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8)

print(frame_clean.info())
print(frame_clean.corrwith(y))
corr_matrix = frame_clean.corr()
corr_matrix.style.background_gradient(cmap='coolwarm')
fig, ax = plt.subplots(figsize=(12, 8))
#corr_matrix.style.background_gradient(cmap='coolwarm')
sns.heatmap(corr_matrix)
print(corr_matrix.columns)
ax.set_xticklabels(frame_clean.columns)
ax.set_yticklabels(frame_clean.columns)
plt.savefig("./images/corr_matrix.jpeg")

plt.show()

frame_clean.corrwith(y).to_csv("./CSV/corr_matrix_with_target.csv")

plt.show()
# DecisionTree, GaussianNB, LinearSVC, MPLClassifier, RandomForest
models_f1= pd.read_csv('f1_dataframe.csv').drop(columns=['Unnamed: 0'])

models_prec=pd.read_csv('prec_dataframe.csv').drop(columns=['Unnamed: 0'])
models_rec=pd.read_csv('rec_dataframe.csv').drop(columns=['Unnamed: 0'])
print(models_rec.describe())

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
X_train2 = StandardScaler().fit_transform(X_train)

feature_names = [f"feature {i}" for i in range(X.shape[1])]
forest = RandomForestClassifier(random_state=0)
forest.fit(X_train2, Y_train)
import time
import numpy as np

start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
import pandas as pd

forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_xticklabels(X_train.columns)
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig("./images/feature_importance.jpeg")
plt.show()

print("f1 score : \n", models_f1.mean())
print("precision score : \n", models_prec.mean())
print("recall score : \n", models_rec.mean())
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(14,10))
ax[0].bar(models_f1.columns, models_f1.mean(), color=['red', 'green', 'blue', 'yellow', 'orange', 'purple'])
ax[1].bar(models_prec.columns, models_prec.mean(),  color=['red', 'green', 'blue', 'yellow', 'orange', 'purple'])
ax[2].bar(models_rec.columns, models_rec.mean(),  color=['red', 'green', 'blue', 'yellow', 'orange', 'purple'])
ax[0].set_ylabel("F1 mean")
ax[1].set_ylabel("Precision mean")
ax[2].set_ylabel("Recall mean")
ax[0].set_title("F1")
ax[1].set_title("Precision")
ax[2].set_title("Recall")
ax[0].set_xticks(models_f1.columns)
ax[1].set_xticks(models_f1.columns)
ax[2].set_xticks(models_f1.columns)


plt.savefig("means_stats.jpeg")
plt.show()



