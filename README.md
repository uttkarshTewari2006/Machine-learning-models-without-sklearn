# Machine Learning Models Without Sklearn

This repository contains implementations of various machine learning models without using the Sklearn library. Each model is implemented from scratch, and the hyperparameters are tuned to achieve optimal performance.

## Models

### Linear Regression
- **Hyperparameters**: Learning rate, number of iterations
- **Description**: Initializes weights array as a zero array and bias as zero, then uses gradient descent `N_iter` times using the provided learning rate to find suitable weights and bias values.
- **Testing**: Tested with the diabetes Sklearn dataset, achieved L1 error of 42.74 and L2 error of 53.93.

### Logistic Regression
- **Hyperparameters**: Learning rate, number of iterations
- **Description**: Same implementation as linear regression, except uses the sigmoid function and thresholding at 0.5 to binary classify data.
- **Testing**: Tested with the breast cancer Sklearn dataset, achieved accuracy of 0.98.

### K-Nearest Neighbors (KNN)
- **Hyperparameters**: Distance type, voting metric
- **Description**: Stores data in the fit phase, finds `N` closest values (using either L1 or L2 distance type) and votes for the classified output (using either weighted average or simple voting).
- **Testing**: Currently testing this model.

### Decision Tree Classifier
- **Hyperparameters**: Information gain type, max depth, min samples split, min samples leaf, max features
- **Description**: Uses the information gain type (Gini impurity or entropy) to find features to split data on, and find the split. Uses other hyperparameters to control complexity and accuracy and avoid overfitting and underfitting.
- **Testing**: Currently working on this model.

### Random Forest Classifier
- **Hyperparameters**: Number of estimators, max depth, min samples split, min samples leaf, max features
- **Description**: Combines multiple decision trees to improve accuracy and control overfitting.
- **Testing**: Currently working on this model.

## How to Use
1. Clone the repository:
-    'git clone git@github.com:uttkarshTewari2006/Machine-learning-models-without-sklearn.git'
3. import the module
-    ex: 'from knn import Knn'
4. initilize the model
-    ex: 'classifier = Knn(distance = "L2", type = "voting")'
5. fit the model
-    ex: 'classifier.fit(numpy_feature_array_train, numpy_label_array_train)'
6. predict the model
-    ex: 'classifier.predict(numpy_feature_aray_test)'
7. check evaluation metrics and tune hyperparameters
