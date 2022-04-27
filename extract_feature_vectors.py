from struct import calcsize
import javalang
import pandas as pd
import numpy as np
import os
input_path = './resources'
array_of_path = []
java_files = []
roots = []


def get_number_of_methods(klass):

    return len(klass.methods)


def get_number_of_fields(klass):
    return len(klass.fields)


def get_number_of_public_methods(methods):
    methods_num=0
    for i in methods:
        #print(i.modifiers)
        if(str(i.modifiers) == "{'public'}"):
            #print(i.modifiers)
            methods_num+=1
    return methods_num

def get_called_methods(methods):
    number_of_called_methods=0
    for i in methods:
        for i, member in i.filter(javalang.tree.MethodInvocation):

            number_of_called_methods+=1
    return number_of_called_methods

def get_number_of_interfaces(klass):
    if((klass.implements)==None):
        return 0
    else: 
        return len(klass.implements)
    


for root, dirs, files in os.walk(input_path, topdown=False):
    #print(f'root: {root} \t dirs: {dirs} \t files: {files}')
    for file in files:
        if(file.endswith(".java")):
            java_files.append(file)
            roots.append(root)
# print(java_files)

classes = []
dict = {"class": [], "MTH": [], "FLD": [], "RFC": [], "INT":[]}
i = 0

for root, file_name in zip(roots, java_files):
    name = os.path.join(root, file_name)

    data = open(name).read()
    tree_tmp = javalang.parse.parse(data)
    # print(tree_tmp.)
    if(i == 0):
        i = 1
        print(tree_tmp.types[0])

    for path, klass in tree_tmp.filter(javalang.tree.ClassDeclaration):
        '''if(isinstance(klass, javalang.tree.ClassDeclaration) and klass.name == file_name.replace('.java', '')):'''

        dict["class"].append(klass.name)
        #print(klass.attrs)
        methods = get_number_of_methods(klass)
        fields = get_number_of_fields(klass)
        public_methods = get_number_of_public_methods(klass.methods)
        interfaces = get_number_of_interfaces(klass)
        called_methods=get_called_methods(klass.methods)
        #print(f"number of called method: {called_methods}")
        #print(f'number of interfaces: {interfaces}')
        #print(f"number of fields: {fields}")
        #print(f"number of method: {methods}")
        dict["MTH"].append(methods)
        dict["FLD"].append(fields)
        dict["RFC"].append(public_methods+ called_methods)
        dict["INT"].append(interfaces)
        # print(klass)
        # print("\n\n\n\n\n\n")

frame = pd.DataFrame(dict)
print(frame)
