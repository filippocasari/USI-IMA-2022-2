from joblib import load
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
frame = pd.read_csv('./CSV/new_feature_vector_file.csv')
'''read the dataset'''
frame_clean = frame.drop(columns=[ 'class','Unnamed: 0'])

y = frame['buggy']
X = frame.drop(columns=['buggy', 'class','Unnamed: 0'])


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



'''
frame_clean.corrwith(y).to_csv("./CSV/corr_matrix_with_target.csv")
plt.savefig("./images/corr_matrix.jpeg")
plt.show()'''
# DecisionTree, GaussianNB, LinearSVC, MPLClassifier, RandomForest
models_f1= pd.read_csv('f1_dataframe.csv')

models_prec=pd.read_csv('prec_dataframe.csv')
models_rec=pd.read_csv('rec_dataframe.csv')
print(models_rec.describe())




