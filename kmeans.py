import csv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from getVectors import getVectors
import numpy as np
from datetime import datetime
import math
np.set_printoptions(threshold=np.nan)

unpack = getVectors()
kapFLC = unpack[0]
kapDates = unpack[1]
lamFLC = unpack[2]
lamDates = unpack[3]

#kapFLC and kapDates for Kappa
#lamFLC and lamDates for Lambda
X = lamFLC
dates = lamDates



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
X_Features = scaler.fit_transform(X)
X_train = X_Features[0:math.floor(X_Features.shape[0]/2)]
X_test = X_Features[math.floor(X_Features.shape[0]/2):X_Features.shape[0]]


kmeans = KMeans(n_clusters=3, n_init=100).fit(X_train)
kmeans.predict(X_test)
kmeans.labels_
kmeans.cluster_centers_
print(kmeans.labels_)
print(kmeans.cluster_centers_)
