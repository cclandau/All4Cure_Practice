# Cece's attempt at using the SciKit learn clustering package
from sklearn import cluster, datasets # http://scikit-learn.org/stable/tutorial/statistical_inference/unsupervised_learning.html
from sklearn.cluster import KMeans
from getVectors import getVectors
import numpy as np

workingMatrix = getVectors()
smallMatrix = workingMatrix[0:850]
# Need to delete MM in row[0], convert slashes to only numerical dates in row[2]
timeKappa = []
timeLambda = []
for row in smallMatrix:











numpyArray = np.array(smallMatrix)
print(numpyArray)
#kmeans = KMeans(n_clusters=5, random_state=0).fit(numpyArray)
k_means = cluster.KMeans(n_clusters=5)
k_means.fit(numpyArray)
print(k_means.labels_[::10])
print()
print(k_means.cluster_centers_)

values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
print()
print("success")

## Want to compare time vs kappa FLC for all
## Want to compare time vs lambda FLC for all
