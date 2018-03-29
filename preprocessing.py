from getTreatments import getTreatments
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
treatDict = getTreatments()

X = np.concatenate((kapFLC, lamFLC), axis=0)
dates = np.concatenate((kapDates, lamDates), axis=0)

#Create first derivative column
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

#Create second derivative column
D2 = np.zeros((len(X), 1))
for i in range(1, len(dates)-1):
    if X[i][0] == X[i-1][0]:
        xdif = dates[i] - dates[i-1]
        ydif = np.float(D1[i]) - np.float(D1[i-1])
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

FLCdict = {}
for i in range(0, X.shape[0]):
	if X[i][0] in FLCdict:
		FLCdict[X[i][0]].append([X[i][1], dates[i], D1[i], D2[i]])
	else:
		FLCdict[X[i][0]] = []
		FLCdict[X[i][0]].append([X[i][1], dates[i], D1[i], D2[i]])

for i in FLCdict:
    if FLCdict[i] !in treatDict:
        del FLCdict[i]

for i in treatDict:
    if treatDict[i] !in FLCdict[i]:
        del treatDict[i]

for i in FLCdict:
	temp = FLCdict[i]
	FLCdict[i] = sorted(temp, key=lambda temp_entry: temp_entry[1])
