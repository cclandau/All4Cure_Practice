import csv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from getVectors import getVectors
from attemptVectorMaker import getKappaLambda
import numpy as np
from datetime import datetime
import math

from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import time
import matplotlib.pyplot as plt

np.set_printoptions(threshold=np.nan)

unpack = getVectors()
kapFLC = unpack[0]
kapDates = unpack[1]
lamFLC = unpack[2]
lamDates = unpack[3]

unpackVectOnly = getKappaLambda()
kappaValues = unpackVectOnly[0]
lambdaValues = unpackVectOnly[1]

# print(kappaValues)
# print()
# print()
print(len(kapDates))
print(len(kappaValues))
plt.subplot(2, 1, 1)
plt.plot(kapDates, kapFLC, 'o-')
plt.title('Kappa FLC')
plt.xlabel('Dates')
plt.ylabel(' KAPPA FLC VALUE')

plt.subplot(2, 1, 2)
plt.plot(lamDates, lamFLC, '.-')
plt.title('Lambda FLC')
plt.xlabel('Dates')
plt.ylabel(' LAMBDA FLC VALUE')
plt.show()
#kapFLC and kapDates for Kappa
#lamFLC and lamDates for Lambda
X = lamFLC
dates = lamDates

print('There are %d data points' % len(X))
#Create and add first derivative column
D1 = np.zeros((len(X), 1))
for i in range(1, len(dates)-1):
    if X[i][0] == X[i-1][0]:
        xdif = dates[i] - dates[i-1]
        ydif = np.float(X[i][1]) - np.float(X[i-1][1])
        if ydif == 0:
            D1[i] = 0
        elif xdif.total_seconds()/86400 == 0:
            if ydif > 0:
                D1[i] = float('inf')
            else:
                D1[i] = float('-inf')
        else:
            D1[i] = ydif/(xdif.total_seconds()/86400)
min_val = D1[np.isfinite(D1)].min()
max_val = D1[np.isfinite(D1)].max()
for i in range(1, len(dates)-1):
    if D1[i] == float('-inf'):
        D1[i] = min_val
    elif D1[i] == float('inf'):
        D1[i] = max_val
X = np.hstack((X, D1))

#Create and add second derivative column
D2 = np.zeros((len(X), 1))
for i in range(1, len(dates)-1):
    if X[i][0] == X[i-1][0]:
        xdif = dates[i] - dates[i-1]
        ydif = np.float(X[i][2]) - np.float(X[i-1][2])
        if ydif == 0:
            D2[i] = 0
        elif xdif.total_seconds()/86400 == 0:
            if ydif > 0:
                D2[i] = float('inf')
            else:
                D2[i] = float('-inf')
        else:
            D2[i] = ydif/(xdif.total_seconds()/86400)
min_val2 = D2[np.isfinite(D2)].min()
max_val2 = D2[np.isfinite(D2)].max()
for i in range(1, len(dates)-1):
    if D2[i] == float('-inf'):
        D2[i] = min_val2
    elif D2[i] == float('inf'):
        D2[i] = max_val2
X = np.hstack((X, D2))

#Deleting the patient identifier
X = np.delete(X, np.s_[0], axis=1)

#Set up training and testing sets 50/50
X = np.array(X, dtype = np.float)
scaler = StandardScaler()
X_Features_scale = scaler.fit_transform(X)
#X_Features_log = np.log(np.absolute(X))
X_train = X_Features_scale[0:math.floor(X_Features_scale.shape[0]/2)]
X_test = X_Features_scale[math.floor(X_Features_scale.shape[0]/2):X_Features_scale.shape[0]]

# ### KMeans Portion ###
# kmeans = KMeans(n_clusters=3, n_init=100).fit(X_train)
# kmeans.predict(X_test)
# kmeans.labels_
# kmeans.cluster_centers_

#logX = open('LogNormalized.csv', 'w')
ScaledX = open('StandardScaler.csv', 'w')
vec = open('vectorWrite.csv', 'w')

# logWrite = csv.writer(logX)
scaleWrite = csv.writer(ScaledX)
vecWrite = csv.writer(vec)

# for row in X_Features_log:
#     logWrite.writerow([row])
for eachRow in X_Features_scale:
    scaleWrite.writerow([eachRow])
for allRows in X:
    vecWrite.writerow([allRows])


#
# ### HDBSCAN Portion ###
# hdb_t1 = time.time()
# hdb = HDBSCAN(min_cluster_size=10).fit(X_Features)
# hdb_labels = hdb.labels_
# hdb_elapsed_time = time.time() - hdb_t1
# n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
#
#
# db_t1 = time.time()
# db = DBSCAN(eps=0.1).fit(X_Features)
# db_labels = db.labels_
# db_elapsed_time = time.time() - db_t1
#
# print('\n\n++ HDBSCAN Results')
# print('Estimated number of clusters: %d' % n_clusters_hdb_)
# print('Silhouette Coefficient: %0.3f'
#       % metrics.silhouette_score(X, hdb_labels))
#
# hdb_unique_labels = set(hdb_labels)
# hdb_colors = plt.cm.Spectral(np.linspace(0, 1, len(hdb_unique_labels)))
# fig = plt.figure(figsize=plt.figaspect(0.5))
# hdb_axis = fig.add_subplot('121')
#
# print('hdbcolors are :')
# print(hdb_colors)
# print('\n\n')
# print(hdb_unique_labels)
#
# for k, col in zip(hdb_unique_labels, hdb_colors):
#     print(k)
#     if k == -1:
#         # Black used for noise.
#         col = 'k'
#     hdb_axis.plot(X[hdb_labels == k, 0], X[hdb_labels == k, 1], 'o', markerfacecolor=col,
#                   markeredgecolor='k', markersize=6)
# hdb_axis.set_title('HDBSCAN\nEstimated number of clusters: %d' % n_clusters_hdb_)
# plt.show()
