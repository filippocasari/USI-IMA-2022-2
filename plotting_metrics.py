import matplotlib.pyplot as plt 
import pandas as pd
import os

path_images = './images'
if(os.path.isdir(path_images) == False):
    os.mkdir(path_images)


models_f1 = pd.read_csv('.f1_dataframe.csv').drop('Unnamed: 0', axis=1)
models_prec = pd.read_csv('.f1_dataframe.csv').drop('Unnamed: 0', axis=1)
models_rec = pd.read_csv('.rec_dataframe.csv').drop('Unnamed: 0', axis=1)

fig_f1 = plt.figure()  
boxplot = models_f1.boxplot()
plt.title("F1_score")
boxplot.set_ylabel('F1')
fig_f1.savefig(path_images+'/f1_plot.png')
plt.show()

fig_prec = plt.figure()  
boxplot2 = models_prec.boxplot()
boxplot2.set_ylabel('Precision')
plt.title("Precision")
fig_prec.savefig(path_images+'/prec_plot.png')
plt.show()

fig_rec = plt.figure()  
boxplot3 = models_rec.boxplot()     
boxplot3.set_ylabel('Recall')
fig_rec.savefig(path_images+'/rec_plot.png')

plt.show()





