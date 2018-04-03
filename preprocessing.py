from getTreatments import getTreatments
from getVectors import getVectors
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
np.set_printoptions(threshold=np.nan)

unpack = getVectors()
kapFLC = unpack[0]
kapDates = unpack[2]
lamFLC = unpack[3]
lamDates = unpack[5]
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

smolderingPatientsDict = {}

for i in FLCdict.keys():
    temp = FLCdict[i]
    FLCdict[i] = sorted(temp, key=lambda temp_entry: temp_entry[1])

keysToDelete = []

with open('unProcessed.csv', 'w') as csvfile:
    temp = []
    for i in FLCdict.keys():
        temp = np.array(FLCdict[i])
        csvfile.write(str(i))
        for j in range (0, temp.shape[0]):
            csvfile.write(", " + str(temp[j][0]))
            csvfile.write(", " + str(temp[j][1]))
        csvfile.write('\n')

for i in FLCdict.keys():
    tempFLC = FLCdict[i]
    if((tempFLC[np.array(FLCdict[i]).shape[0] - 1][1] - tempFLC[0][1]).days <= 180):
        print("patient with less than six months: " + i)
        keysToDelete.append(i)
    else:
        if i not in treatDict.keys():
            smolderingPatientsDict[i] = FLCdict[i]
            keysToDelete.append(i)
            #print("smoldering patient: " + i)
        else:
            tempTreat = treatDict[i]
            if(tempFLC[0][1] > tempTreat[0][1]):
                keysToDelete.append(i)
                print("patient with treatment before reading: " + i)
            haveFoundSixMonth = False
            for j in range(0, np.array(FLCdict[i]).shape[0]): #for every row in matrix
                if (((tempFLC[j][1] - tempFLC[0][1]).days >= 180) and (haveFoundSixMonth != True)):
                    sixMonthIndex = j
                    haveFoundSixMonth = True
            firstSixMonths = np.array(tempFLC)[:(sixMonthIndex), :]
            FLCdict[i] = firstSixMonths
            tempFLC = FLCdict[i]
            #print("patient: " + i)
            #print(tempFLC[:, 1])
            #else:
                #print("good patient: " + i)


for i in keysToDelete:
    del FLCdict[i]

with open('processed.csv', 'w') as csvfile:
    temp = []
    for i in FLCdict.keys():
        temp = np.array(FLCdict[i])
        if(temp.shape[0]>=5):
            csvfile.write(str(i))
            for j in range (0, 5):
                csvfile.write(", " + str(temp[j][0]))
                csvfile.write(", " + str(temp[j][1]))
            csvfile.write('\n')

print(len(FLCdict.keys()))




'''for i in FLCdict.keys():
   tempFLC = np.array(FLCdict[i])
   plt.figure()
   plt.plot(tempFLC[:, 1], tempFLC[:, 0])
   plt.title(i)
   # print(tempFLC[:, 0])
   # print(tempFLC[:, 1])
   plt.show()'''
