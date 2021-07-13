import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# importing the Dataset
dataset = pd.read_csv('iris.xls', names=['sepal length', 'sepal width', 'petal length', 'petal width', 'Target'])
dataset.head()

# Here We import the needed library for classification task"
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split  # Import train_test_split function
from sklearn import metrics  # Import scikit-learn metrics module for accuracy calculation

# Getting the four features in data variable and target, which our class label in target vriable from iris dataset
data = dataset.loc[:, ['sepal length', 'sepal width', 'petal length', 'petal width']]
target = dataset.loc[:, 'Target']
data.head()

# Splitting data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=1)

# Creating a Decision Tree Model
Tree_model = DecisionTreeClassifier()

# we fit our model using our training data.
Tree_model.fit(x_train, y_train)

# Then we score the predicted output from model on our test data against our ground truth test data.
y_predict_tree = Tree_model.predict(x_test)
Accuracy = metrics.accuracy_score(y_test, y_predict_tree)
print("the accuracy for the decision tree is:", round(Accuracy, 2))

# Building a KNN model
Knn_model = KNeighborsClassifier(n_neighbors=5)
Knn_model.fit(x_train, y_train)
y_predict_Knn = Knn_model.predict(x_test)
score = metrics.accuracy_score(y_test, y_predict_Knn)

# the accuracy of the KNN model with n_neighbors equals to 5
print("the accuracy for the KNN with k=5 is:", round(score, 2))

# Building a KNN model with multiple values of k
k_range = range(1, 40)
scores_list = list()
for k in k_range:
    Knn_model = KNeighborsClassifier(n_neighbors=k)
    Knn_model.fit(x_train, y_train)
    y_predict_Knn = Knn_model.predict(x_test)
    score = metrics.accuracy_score(y_test, y_predict_Knn)
    scores_list.append(score)

# plotting the Accuracy for each K
plt.plot(k_range, scores_list)
plt.xlabel('value of K for the KNN model')
plt.ylabel('Testing Accuracy')
plt.show()


# Comparison between KNN and Decision trees

# KNN is unsupervised, Decision Tree (DT) supervised.

# the KNN model is better, although changing random_state for the test/train data will produce some
# scores equals for both models.
# KNN determines neighborhoods, so there must be a distance metric.
# This implies that all features must be numeric.

# Distance metrics may be effected by varying scales between attributes and also high-dimensional space.

# DT on the other hand predicts a class for a given input vector. The attributes may be numeric or nominal.

# the Decision tree is faster due to KNN’s expensive real time execution.
