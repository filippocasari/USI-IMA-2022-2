import matplotlib.pyplot as plt 
import pandas as pd

models_f1 = pd.read_csv('.f1_dataframe.csv').drop('Unnamed: 0', axis=1)
models_prec = pd.read_csv('.f1_dataframe.csv').drop('Unnamed: 0', axis=1)
models_rec = pd.read_csv('.rec_dataframe.csv').drop('Unnamed: 0', axis=1)
boxplot = models_f1.boxplot()
boxplot.set_ylabel('f1')
plt.show()
boxplot2 = models_prec.boxplot()   
boxplot2.set_ylabel('precision')
plt.show()
boxplot3 = models_rec.boxplot()     
boxplot3.set_ylabel('recall')
plt.show()