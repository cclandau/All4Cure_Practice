import csv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from getVectors import getVectors
import numpy as np
from datetime import datetime

matrix = getVectors()
smallMatrix = matrix[0:850]
X = np.array(smallMatrix)

#Create vector of datetimes to find difference between dates
dates = []
for i in range(0, 850):
    dates.append(datetime.strptime(X[i][3], '%m%d%Y').date())

#Delete original date column because it is no longer needed
X = np.delete(X, np.s_[3], axis=1)

#Create and add first derivative column
X = np.array(X, dtype = np.float)
D1 = np.zeros((len(X), 1))
for i in range(1, 850):
    if X[i][0] == X[i-1][0]:
        xdif = dates[i] - dates[i-1]
        ydif = X[i][1] - X[i-1][1]
        if ydif == 0:
            D1[i] = 0
        else:
            D1[i] = ydif/(xdif.total_seconds()/86400)
min_val = D1[np.isfinite(D1)].min()
max_val = D1[np.isfinite(D1)].max()
for i in range(1, 850):
    if D1[i] == float('-inf'):
        D1[i] = min_val
    elif D1[i] == float('inf'):
        D1[i] = max_val
X = np.hstack((X, D1))

#Create and add second derivative column
D2 = np.zeros((len(X), 1))
for i in range(1, 850):
    if X[i][0] == X[i-1][0]:
        xdif = dates[i] - dates[i-1]
        ydif = X[i][3] - X[i-1][3]
        if ydif == 0:
            D2[i] = 0
        else:
            D2[i] = ydif/(xdif.total_seconds()/86400)
min_val2 = D2[np.isfinite(D2)].min()
max_val2 = D2[np.isfinite(D2)].max()
for i in range(1, 850):
    if D2[i] == float('-inf'):
        D2[i] = min_val2
    elif D2[i] == float('inf'):
        D2[i] = max_val2
X = np.hstack((X, D2))

#Deleting the patient identifier to see if the clustering is better
X = np.delete(X, np.s_[0], axis=1)
#Deleting lambda values to fit how our vectors will be once getVectors is updated
X = np.delete(X, np.s_[1], axis=1)

scaler = StandardScaler()
X_Features = scaler.fit_transform(X)
X_train = X_Features[0:425]
X_test = X_Features[426:850]
kmeans = KMeans(n_clusters=3, n_init=100).fit(X_train)
kmeans.predict(X_test)
kmeans.labels_
kmeans.cluster_centers_
print(kmeans.labels_)
print(kmeans.cluster_centers_)
for x in range(426, 850):
    print(X_Features[x])
