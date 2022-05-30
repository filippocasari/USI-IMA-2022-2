from joblib import load

path = "./MODELS/"
extension = ".joblib"
dt = load(path + "DecisionTree"+extension)
print(dt)
dt = load(path + "GaussianNB"+extension)
print(dt)
dt = load(path + "LinearSVC"+extension)
print(dt)
dt = load(path + "MPLClassifier"+extension)
print(dt)
dt = load(path + "RandomForestClassifier"+extension)
print(dt)