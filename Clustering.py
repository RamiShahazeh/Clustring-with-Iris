import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn import metrics

# importing the Dataset
dataset = pd.read_csv('iris.xls', names=['sepal length', 'sepal width', 'petal length', 'petal width', 'Target'])

# Getting the four features in data variable and target, which our class label in target variable from iris Dataset
data = dataset.loc[:, ['sepal length', 'sepal width', 'petal length', 'petal width']]
# the target may be [0,1,2] or [1,2,3]
# we just want to make it fit with the K-mean results
# which gives a results in [0,1,2]
target = dataset.loc[:, 'Target'] - 1

x = pd.DataFrame(data)
x.columns = ['Sepal_Length', 'Sepal_Width', 'Petal_Length', 'Petal_Width']

y = pd.DataFrame(target)
y.columns = ['Targets']

# K Means Cluster
model = KMeans(n_clusters=3, random_state=1)
model.fit(x)

# This is what KMeans thought
model.labels_

# View the results
# Set the size of the plot
plt.figure(figsize=(14, 7))

# Create a colormap
colormap = np.array(['red', 'lime', 'black'])

# Plot the Original Classifications
plt.subplot(1, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')

# Plot the Models Classifications
plt.subplot(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[model.labels_], s=40)
plt.title('K Mean Classification (before relabel)')
# plt.show()
# we compare both of them and we will reorder the labels
relabel = np.choose(model.labels_, [1, 0, 2]).astype(np.int64)

plt.figure(figsize=(14, 7))

# Create a colormap
colormap = np.array(['red', 'lime', 'black'])

# Plot the Original Classifications
plt.subplot(1, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')

# Plot the Models Classifications
plt.subplot(1, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[relabel], s=40)
plt.title('K Mean Classification (after relabel)')
# plt.show()

print(classification_report(y, relabel))
# we want Hige precision and high recall
# we see that in label '0' there is 100% in both recall and precision

# silhouette coefficients for k-means
sil_Score = metrics.silhouette_score(x, model.labels_, metric='euclidean')
print("The Silhouette Coefficient for k-means= ", sil_Score)

# Agglomerative Hierarchical Clustering


# Agglomerative Hierarchical Clustering using Single linkage
clusterSingle = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='single')
clusterSingle.fit_predict(data)

# Agglomerative Hierarchical Clustering using complete linkage
clusterComplete = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='complete')
clusterComplete.fit_predict(data)

# Agglomerative Hierarchical Clustering using average linkage
clusterAverage = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
clusterAverage.fit_predict(data)

# plotting the clusters
plt.figure(figsize=(14, 7))
plt.subplot(1, 3, 1)
plt.title("Single linkage")
plt.scatter(x.Petal_Length, x.Petal_Width, c=clusterSingle.labels_, cmap='rainbow')

plt.subplot(1, 3, 2)
plt.title("complete linkage")
plt.scatter(x.Petal_Length, x.Petal_Width, c=clusterComplete.labels_, cmap='rainbow')

plt.subplot(1, 3, 3)
plt.title("average linkage")
plt.scatter(x.Petal_Length, x.Petal_Width, c=clusterAverage.labels_, cmap='rainbow')

# validate the hierarchical clusters results using External index validation method
plt.figure(figsize=(14, 7))

# Create a colormap
colormap = np.array(['red', 'lime', 'black'])

# Plot the Original Classifications
plt.subplot(2, 2, 1)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[y.Targets], s=40)
plt.title('Real Classification')

# Plot the Models Classifications
plt.subplot(2, 2, 2)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[clusterSingle.labels_], s=40)
plt.title('single linkage')

# Plot the Models Classifications
plt.subplot(2, 2, 3)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[clusterComplete.labels_], s=40)
plt.title('complete linkage')

# Plot the Models Classifications
plt.subplot(2, 2, 4)
plt.scatter(x.Petal_Length, x.Petal_Width, c=colormap[clusterAverage.labels_], s=40)
plt.title('avarage linkage')

# silhouette coefficients for single, complete, average linkage
sil_Score1 = metrics.silhouette_score(x, clusterSingle.labels_, metric='euclidean')
print("The Silhouette Coefficient for single linkage = ", sil_Score1)
sil_Score2 = metrics.silhouette_score(x, clusterComplete.labels_, metric='euclidean')
print("The Silhouette Coefficient for Complete linkage = ", sil_Score2)
sil_Score3 = metrics.silhouette_score(x, clusterAverage.labels_, metric='euclidean')
print("The Silhouette Coefficient for Average linkage = ", sil_Score3)

# print(classification_report(y, clusterSingle.labels_))
# print(classification_report(y, clusterComplete.labels_))
# print(classification_report(y, clusterAverage.labels_))
plt.show()

# im my opinion for this Dataset applying k-means as we provided after relabel the model of k-means we see that the
# precision and accuracy of the model are really good. another reason why we chose k-means The one and the most basic
# difference is where to use K means and Hierarchical clustering is on the basis of Scalability and Flexibility.
# Hierarchical is Flexible but can not be used on large data. K means is scalable but cannot use for flexible data.
