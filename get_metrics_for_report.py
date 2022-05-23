import pandas as pd

feature_vectors = pd.read_csv("./CSV/feature_vectors.csv")
print(feature_vectors.mean())
print()
print(feature_vectors.min())
print()
print(feature_vectors.max())
print()
print(feature_vectors.std())
feature_vectors_new = pd.read_csv("./CSV/new_feature_vector_file.csv")
non_buggy_classes = len(
    feature_vectors_new.loc[feature_vectors_new['buggy'] == 0])
buggy_classes = len(feature_vectors_new.loc[feature_vectors_new['buggy'] == 1])
print(f"Number of non buggy classes:  {non_buggy_classes}\n Number of buggy classes:  {buggy_classes}\n")
