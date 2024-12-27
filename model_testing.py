from knn import Knn
from logistic_regression import logistic_regression
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from desision_tree import DecisionTree
from random_forest import RandomForest
from knn import Knn

def createClf(clf, args = {}):
    if (clf == "knn"):
        return Knn(n = args.get("n", 3), distance = args.get("distance", "L2"), type = args.get("type", "voting"))
    if (clf == "lr"):
        return logistic_regression(lr = args.get("lr", 0.01), n_iter = args.get("n_iter", 1000))
    if (clf == "dt"):
        return DecisionTree(max_depth = args.get("max_depth", None), min_samples_split= args.get("min_samples_split", 2))
    if (clf == "rf"):
        return RandomForest(n_trees = args.get("n_trees", 10), max_depth = args.get("max_depth", None), min_sample_split = args.get("min_samples_split", 2), type = args.get("type", "voting"), ignore_index = args.get("ignore_index", []), sample_percent = args.get("sample_percent", 0.7))
    print(clf)
    raise Exception("ValueError: incorrect clf type")
    
def kfold(clf_str, X, Y, args = {}, k_folds = 5):
    num_samples = X.shape[0]
    accuracies = []
    indices = np.arange(num_samples)
    fold_size = num_samples // k_folds

    for i in range(k_folds):
        test_indices = indices[i * fold_size : (i + 1) * fold_size]
        train_indices = np.setdiff1d(indices, test_indices)

        X_train, Y_train = X[train_indices], Y[train_indices]
        X_test, Y_test = X[test_indices], Y[test_indices]

        clf = createClf(clf_str, args)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)

        accuracies.append(np.sum(Y_pred == Y_test) / len(Y_test))
    return np.mean(accuracies)

    

breast_cancer = datasets.load_breast_cancer()

X = breast_cancer["data"]
Y = breast_cancer["target"]

scalar = StandardScaler()
X = scalar.fit_transform(X)

knn_accuracy = kfold("knn", X, Y, args = {"n": 5})
lr_accuracy = kfold("lr", X, Y)
dt_accuracy = kfold("dt", X, Y, args = {"max_depth": 3, "min_samples_split": 5})
rf_accuracy = kfold("rf", X, Y, args={"n_trees": 15, "max_depth": 4, "min_samples_split": 4})

print("knn accuracy with breast cancer: ", knn_accuracy)
print("logistic regression accuracy with breast cancer: ", lr_accuracy)
print("decision tree accuracy with breast cancer: ", dt_accuracy)
print("random forest tree accuracy with breast cancer: ", rf_accuracy)


iris = datasets.load_iris()

X = iris["data"]
Y = iris["target"]

'''
knn_accuracy = kfold("knn", X, Y, args = {"n": 3})
lr_accuracy = kfold("lr", X, Y, args={"lr": 0.001})
dt_accuracy = kfold("dt", X, Y, args = {"max_depth": 5})

print("knn accuracy with iris: ", knn_accuracy)
print("logistic regression accuracy with iris: ", lr_accuracy)
print("decision tree accuracy with iris: ", dt_accuracy)'''