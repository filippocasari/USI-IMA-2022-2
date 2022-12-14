from extract_feature_vectors import return_first_step
import os
import pandas as pd
import numpy as np
import time
import sys


def read_content(file):
    '''read the content of a .src file. \
        Returns the np.array of strings contained in the file'''
    # print(file)
    f = open(file)

    string_classes = f.read().split('\n')
    new_strings = []

    for i in string_classes[:-1]:
        classes = i
        while(classes.find(".") != -1):
            # print(classes)
            classes = classes[classes.find('.')+1:]
        new_strings.append(classes)
    new_strings = np.array(new_strings)

    return new_strings

def second_step(write_csv=False):
    """ Label feature vectors, second step of the project

    Args:
        write_csv (bool, optional): Do u wanna write the csv file?. Defaults to False.
    """    ''''''
    start = time.time()
    '''start execution time'''
    buggy_classes = []
    '''buggy classes array'''
    for root, dirs, files in os.walk(buggy_class_path, topdown=False):
        #print(f'root: {root} \t dirs: {dirs} \t files: {files}')
        for file in files:
            if(file.endswith(".src")):
                buggy_classes_files.append(file)
                roots.append(root)
                strings = read_content(root + file)
                # print(strings)
                for i in strings:
                    buggy_classes.append(i)

    buggy_classes = np.array(buggy_classes)
    print(f"number of buggy classes found in the file: {len(buggy_classes)}")
    # buggy_classes=buggy_classes.flatten()

    # print(buggy_classes)
    # frame['buggy']=np.zeros(frame.shape[0])
    classes = frame['class']
    '''taking classes from the first column'''
    # print(classes)
    dictionary = {'buggy': []}
    '''dictionary'''
    for i in classes:

        if(i in buggy_classes):
            '''set to 1 if this class is buggy'''
            dictionary['buggy'].append(1)
        else:
            '''otherwise'''
            dictionary['buggy'].append(0)


    frame['buggy'] = dictionary['buggy']
    '''adding new column'''
    print("How many buggy classes found? ",
        frame.loc[frame['buggy'] == 1].shape[0])
        
    print(frame)
    path_csv = './CSV/'
    '''path for csv files'''
    
    if(write_csv):
        '''write the csv of second step'''
        frame.to_csv(path_csv+'new_feature_vector_file.csv')
    end = time.time()
    '''end execution'''
    print(f"time execution second step: {end-start}")

#print(f"Buggy classes {buggy_classes}")

write_csv = False

print("Arguments: "+str((sys.argv[1:])))
if (len(sys.argv) ==1 ):
    print("using default values for this step")
elif(len(sys.argv) >=2 ):
    
    print("Writing Csv: " + (sys.argv[1]))
    if((sys.argv[1]) == 'True'):
        write_csv = True
        
    print("Executing step one first: " + ((sys.argv[2])))
    if(sys.argv[2] == 'True'):
        
        frame=return_first_step(writing_csv=True)
    else:
        
        print("starting second step, reading file from path: './CSV/feature_vectors.csv' ")
        frame = pd.read_csv('./CSV/feature_vectors.csv')
        '''read frame from csv dir'''
    
else:
    sys.exit("exiting 'cause sys arguments more than 1 or 2")



# print(frame.head())
buggy_class_path = './resources/modified_classes/'
'''path for baggy classes'''
buggy_classes_files = []
'''names for buggy files'''
roots = []
'''root of buggy files'''

second_step(write_csv=write_csv)

