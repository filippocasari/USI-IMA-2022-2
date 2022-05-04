import time
import re
from struct import calcsize
import javalang
import pandas as pd
import numpy as np
import os
input_path = './resources/defects4j-checkout-closure-1f'
array_of_path = []
java_files = []
roots = []


def get_number_of_methods(klass):

    return len(klass.methods)


def get_statements(klass):
    max = 0
    number_statements = 0.
    for method in klass.methods:
        j = 0
        for i in method.filter(javalang.tree.Statement):
            j += 1
            number_statements += 1.
        if(max < j):
            max = j
    return max, number_statements


def get_number_of_fields(klass):
    return len(klass.fields)


def get_number_of_public_methods(methods):
    methods_num = 0
    for i in methods:
        # print(i.modifiers)
        if(str(i.modifiers) == "{'public'}"):
            # print(i.modifiers)
            methods_num += 1
    return methods_num


def get_called_methods(methods):
    number_of_called_methods = 0
    for i in methods:
        for i, member in i.filter(javalang.tree.MethodInvocation):

            number_of_called_methods += 1
    return number_of_called_methods


def get_number_of_interfaces(klass):
    if((klass.implements) == None):
        return 0
    else:
        return len(klass.implements)


def get_cpx(klass):

    max = 0
    for method in klass.methods:
        j = 0
        for i in method.filter(javalang.tree.IfStatement):
            j += 1
        for i in method.filter(javalang.tree.SwitchStatement):
            j += 1
        for i in method.filter(javalang.tree.WhileStatement):
            j += 1
        for i in method.filter(javalang.tree.ForStatement):
            j += 1
        for i in method.filter(javalang.tree.DoStatement):
            j += 1
        if(max < j):
            max = j
    return max


def get_return(klass):

    max = 0
    for method in klass.methods:
        j = 0
        for i in method.filter(javalang.tree.ReturnStatement):
            j += 1

        if(max < j):
            max = j
    return max


def get_exceptions(klass):

    max = 0
    for method in klass.methods:
        j = 0
        for i in method.filter(javalang.tree.ThrowStatement):
            j += 1

        if(max < j):
            max = j
    return max


def get_block(klass):
    count_doc_class = 0
    count_word_sub_max = 0
    counter_words_comments = 0
    if(klass.documentation != None):
        count_doc_class = 1
        s = klass.documentation
        sub = re.findall('\w+', s)
        count_word_sub_max = len(sub)
        counter_words_comments += len(s.split())

    # print(len(klass.documentation))

    for method in klass.methods:
        if(method.documentation != None):
            count_doc_class += 1
            s = method.documentation
            counter_words_comments += len(s.split())
            sub = re.findall('\w+', s)
            if(count_word_sub_max < len(sub)):
                count_word_sub_max = len(sub)

    return count_doc_class, count_word_sub_max, counter_words_comments


def get_length_name_methods(klass):
    array_methods_name = []
    for method in klass.methods:
        array_methods_name.append(len(method.name))
    if(len(array_methods_name) == 0):
        return 0.
    array_methods_name = np.array(array_methods_name)
    # print(array_methods_name)
    return np.mean(array_methods_name)


# use re.findall('\w+', s) to get all alphanumeric substrings in s
def return_first_step(writing_csv=True):
    import time
    start=time.time()
    for root, dirs, files in os.walk(input_path, topdown=False):
        #print(f'root: {root} \t dirs: {dirs} \t files: {files}')
        for file in files:
            if(file.endswith(".java")):
                java_files.append(file)
                roots.append(root)
    # print(java_files)

    
    dict = {"class": [], "MTH": [], "FLD": [], "RFC": [],
            "INT": [], "SZ": [], "CPX": [], "EX": [], "RET": [], "BCM": [], "NML": [], "WRD": [], "DCM": []}
    

    for root, file_name in zip(roots, java_files):
        name = os.path.join(root, file_name)

        data = open(name).read()
        tree_tmp = javalang.parse.parse(data)
        # print(tree_tmp.)

        for path, klass in tree_tmp.filter(javalang.tree.ClassDeclaration):
            '''if(isinstance(klass, javalang.tree.ClassDeclaration) and klass.name == file_name.replace('.java', '')):'''
            if(klass.name != file_name.replace('.java', '')):
                continue
            dict["class"].append(klass.name)
            # print(klass.attrs)
            methods = get_number_of_methods(klass)
            fields = get_number_of_fields(klass)
            public_methods = get_number_of_public_methods(klass.methods)
            interfaces = get_number_of_interfaces(klass)
            called_methods = get_called_methods(klass.methods)
            max_statements, number_statements = get_statements(klass)
            max_cpx = get_cpx(klass)
            max_return = get_return(klass)
            max_ex = get_exceptions(klass)
            blocks, count_word_sub_max, counter_words_comments = get_block(klass)
            average_length_names_methods = get_length_name_methods(klass)
            # print(max_cpx)
            #print(f"number of called method: {called_methods}")
            #print(f'number of interfaces: {interfaces}')
            #print(f"number of fields: {fields}")
            #print(f"number of method: {methods}")
            dict["MTH"].append(methods)
            dict["FLD"].append(fields)
            dict["RFC"].append(public_methods + called_methods)
            dict["INT"].append(interfaces)
            dict["SZ"].append(max_statements)
            dict["CPX"].append(max_cpx)
            dict["RET"].append(max_return)
            dict["EX"].append(max_ex)
            dict["BCM"].append(blocks)

            dict["NML"].append(round(average_length_names_methods,3))
            dict["WRD"].append(count_word_sub_max)
            if(number_statements == 0):
                dict["DCM"].append(0.)
            else:
                dict["DCM"].append(round(counter_words_comments/number_statements, 3))

            # print(klass)
            # print("\n\n\n\n\n\n")

    end = time.time()
    print(f"first step execution time: {end-start}")
    frame = pd.DataFrame(dict)
    #print(frame.sort_values('class').head())
    #print(frame)

    path_csv = 'CSV/'
    if(writing_csv):
        if(os.path.isdir(path_csv) == False):
            os.mkdir(path_csv)
        frame.to_csv(path_csv+'feature_vectors.csv')
    return frame

return_first_step(False)

