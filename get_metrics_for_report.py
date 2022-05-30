import pandas as pd

feature_vectors = pd.read_csv("./CSV/feature_vectors.csv").drop(columns=['Unnamed: 0', 'class'])
columns = feature_vectors.columns


print("MEAN\n")
means= feature_vectors.mean().round(3)
print(feature_vectors.mean())
mins = feature_vectors.min()
print("MIN\n")
print(feature_vectors.min())
print("MAX\n")
maxs = feature_vectors.max()
print(feature_vectors.max())
print("STD\n")
stds = feature_vectors.std().round(3)
print(feature_vectors.std())
feature_vectors_new = pd.read_csv("./CSV/new_feature_vector_file.csv")
non_buggy_classes = len(
    feature_vectors_new.loc[feature_vectors_new['buggy'] == 0])
buggy_classes = len(feature_vectors_new.loc[feature_vectors_new['buggy'] == 1])
print(f"Number of non buggy classes:  {non_buggy_classes}\n Number of buggy classes:  {buggy_classes}\n")


getting_metrics = pd.DataFrame([columns, means, mins, maxs, stds])
print(getting_metrics)
getting_metrics.to_csv("./CSV/methods_stats.csv")
