from extract_feature_vectors import return_first_step
import os
import pandas as pd
import numpy as np
import time
#frame = return_first_step(False)
frame = pd.read_csv('./CSV/feature_vectors.csv')
#print(frame.head())
buggy_class_path='./resources/modified_classes/'
buggy_classes_files=[]
roots=[]



def read_content(file):
    #print(file)
    f = open(file)

    string_classes=f.read().split('\n')
    new_strings=[]
    
    for i in string_classes[:-1]:
        classes=i
        while(classes.find(".")!= -1):
            #print(classes)
            classes=classes[classes.find('.')+1:]
        new_strings.append(classes)
    new_strings=np.array(new_strings)
    
    
    return new_strings

start= time.time()
buggy_classes=[]
for root, dirs, files in os.walk(buggy_class_path, topdown=False):
        #print(f'root: {root} \t dirs: {dirs} \t files: {files}')
        for file in files:
            if(file.endswith(".src")):
                buggy_classes_files.append(file)
                roots.append(root)
                strings=read_content(root + file)
                #print(strings)
                for i in strings:
                    buggy_classes.append(i)
buggy_classes=np.array(buggy_classes)
print(f"number of buggy classes found in the file: {len(buggy_classes)}")
#buggy_classes=buggy_classes.flatten()

#print(buggy_classes)
#frame['buggy']=np.zeros(frame.shape[0])
classes = frame['class']
#print(classes)
dictionary = {'buggy':[]}
for i in classes:
    
    if(i in buggy_classes):
        dictionary['buggy'].append(1)
    else:
        dictionary['buggy'].append(0)


frame['buggy']= dictionary['buggy']
print("How many buggy classes found? ", frame.loc[frame['buggy']==1].shape[0])
print(frame)
path_csv = './CSV/'
write_csv=True
if(write_csv):
    frame.to_csv(path_csv+'new_feature_vector_file.csv')
end=time.time()
print(f"time execution second step: {end-start}")








#print(f"Buggy classes {buggy_classes}")