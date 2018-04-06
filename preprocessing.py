from getTreatments import getTreatments
from getVectors import getVectors
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import math
from scipy import stats
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.datasets.samples_generator import make_blobs
import time
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

#! Not really necessary
with open('unProcessed.csv', 'w') as csvfile:
    temp = []
    for i in FLCdict.keys():
        temp = np.array(FLCdict[i])
        csvfile.write(str(i))
        for j in range (0, temp.shape[0]):
            csvfile.write(", " + str(temp[j][0]))
            csvfile.write(", " + str(temp[j][1]))
        csvfile.write('\n')
#!

for i in FLCdict.keys():
    tempFLC = FLCdict[i]
    if((tempFLC[np.array(FLCdict[i]).shape[0] - 1][1] - tempFLC[0][1]).days <= 180):
        #print("patient with less than six months: " + i)
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
                #print("patient with treatment before reading: " + i)
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

preSpearman = []
useablePatients = []
with open('processed.csv', 'w') as csvfile:
    temp = []
    for i in FLCdict.keys():
        temp = np.array(FLCdict[i])
        if(temp.shape[0]>=5): # will need to change 5 to a field
            temp2 = [temp[0][0], temp[1][0], temp[2][0], temp[3][0], temp[4][0]]
            csvfile.write(str(i))
            useablePatients.append(str(i))
            for j in range (0, 5):
                csvfile.write(", " + str(temp[j][0]))
                csvfile.write(", " + str(temp[j][1]))
            csvfile.write('\n')
            preSpearman.append(temp2)
preSpearman = np.array(preSpearman)
preSpearmanNum = np.array(preSpearman.astype(float))

unprocessedMatrix = np.array(list(zip(useablePatients, preSpearmanNum)), dtype=object)
with open('unscaledData.csv', 'w') as csvfile:
    csvfile.write("Patient Number + UnScaled FLC Values" + '\n')
    csvfile.write('\n'.join('{}, {}'.format(x[0], x[1]) for x in unprocessedMatrix))

### creating the spearman's correlation matrix ###
spearman = np.zeros((preSpearman.shape[0], preSpearman.shape[0]))
for i in range(0, preSpearman.shape[0]):
    for j in range(0, preSpearman.shape[0]):
        SpearR = stats.spearmanr(preSpearman[i, :], preSpearman[j, :])
        corr = SpearR[0]
        spearman[i, j] = corr
        spearman[j, i] = corr

spearman = np.round(spearman, 1)
np.savetxt("spearman.csv", spearman, delimiter=",")

### plotting flc value for each patient ###
# for i in FLCdict.keys():
#    tempFLC = np.array(FLCdict[i])
#    plt.figure()
#    plt.plot(tempFLC[:, 1], tempFLC[:, 0])
#    plt.title(i)
#    # print(tempFLC[:, 0])
#    # print(tempFLC[:, 1])
#    plt.show()

# ### HDBSCAN Portion ###
hdb_t1 = time.time()
hdb = HDBSCAN(min_cluster_size=2).fit(preSpearman)
hdb_labels = hdb.labels_
hdb_prob = hdb.probabilities_
hdb_elapsed_time = time.time() - hdb_t1
n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
merged = np.array(list(zip(useablePatients, hdb_labels, hdb_prob)))


with open('hdbscanPairs.csv', 'w') as csvfile:
    csvfile.write("Patient Number, Cluster Label, Cluster Probability" + '\n')
    csvfile.write('\n'.join('{}, {}, {}'.format(x[0], x[1], x[2]) for x in merged))


print('\n\nHDBSCAN Results')
print('Estimated number of clusters: %d' % n_clusters_hdb_)
print('Patients, Cluster Labels, and Probability')
print(merged)

### minmax Normalization ###
minValues = preSpearmanNum.min(axis=1)
minMaxMatrix = np.zeros((len(minValues), 5))
index = 0;
row = 0;
#print("FLC Value of each element, Index, Row, minValue and maxValue")
for patient in np.nditer(preSpearmanNum):
    #print("'{0}', '{1}', '{2}', '{3}'".format(patient, index, row, minValues[row]))
    temp = np.array(patient)
    temp = temp - minValues[row] # will need to change to be a field
    # Subtracting original min portion
    minMaxMatrix[row, index] = temp
    index = index + 1
    # in the last column, need to determine max value of new row & update
    if (index == 5):
        maxVal = minMaxMatrix[row, :].max()
        minMaxMatrix[row, :] = minMaxMatrix[row, :] / maxVal
        index = 0
        row = row + 1
normalizedWithPatientNum = np.array(list(zip(useablePatients, minMaxMatrix)), dtype=object)

with open('minMaxNormalized.csv', 'w') as csvfile:
    csvfile.write("Patient Number + Normalized FLC Values")
    csvfile.write('\n')
    csvfile.write('\n'.join('{}, {}'.format(x[0], x[1]) for x in normalizedWithPatientNum))
### hdb minmax normalized ###
hdb_t1_mm = time.time()
hdb_mm = HDBSCAN(min_cluster_size=2).fit(minMaxMatrix)
hdb_labels_mm = hdb_mm.labels_
hdb_prob_mm = hdb_mm.probabilities_
hdb_elapsed_time_mm = time.time() - hdb_t1_mm
n_clusters_hdb_mm = len(set(hdb_labels_mm)) - (1 if -1 in hdb_labels_mm else 0)
merged_mm = np.array(list(zip(useablePatients, hdb_labels_mm, hdb_prob_mm)))

with open('hdbscanPairs_minmax.csv', 'w') as csvfile:
    csvfile.write("Patient Number, Cluster Label, Cluster Probability" + '\n')
    csvfile.write('\n'.join('{}, {}, {}'.format(x[0], x[1], x[2]) for x in merged_mm))

print('\n\nHDBSCAN Results for Min-Max Normalization')
print('Estimated number of clusters: %d' % n_clusters_hdb_mm)
print('Patients, Cluster Labels, and Probability')
print(merged_mm)

### Log Normalization portion ###
#logRawMatrix = np.log10(preSpearman.astype(float) + 1)
logRawMatrix = np.log2(preSpearman.astype(float) + 1)

print(np.log2(preSpearman.astype(float) + 1))
print()
print(np.log10(preSpearman.astype(float) + 1))
logScaledWithPatientNum = np.array(list(zip(useablePatients, logRawMatrix)), dtype=object)

with open('log2Scaled.csv', 'w') as csvfile:
    csvfile.write("Patient Number + Normalized FLC Values")
    csvfile.write('\n')
    csvfile.write('\n'.join('{}, {}'.format(x[0], x[1]) for x in logScaledWithPatientNum))

### hdb log scaled ###
hdb_t1_log = time.time()
hdb_log = HDBSCAN(min_cluster_size=2).fit(logRawMatrix)
hdb_labels_log = hdb_log.labels_
hdb_prob_log = hdb_log.probabilities_
hdb_elapsed_time_log = time.time() - hdb_t1_log
n_clusters_hdb_log = len(set(hdb_labels_log)) - (1 if -1 in hdb_labels_log else 0)
merged_log = np.array(list(zip(useablePatients, hdb_labels_log, hdb_prob_log)))

with open('hdbscanPairs_log2.csv', 'w') as csvfile:
    csvfile.write("Patient Number, Cluster Label, Cluster Probability" + '\n')
    csvfile.write('\n'.join('{}, {}, {}'.format(x[0], x[1], x[2]) for x in merged_log))

print('\n\nHDBSCAN Results for Log Scaling')
print('Estimated number of clusters: %d' % n_clusters_hdb_log)
print('Patients, Cluster Labels, and Probability')
print(merged_log)

### hdb ___ plotting ###
# hdb_unique_labels = set(hdb_labels)
# hdb_colors = plt.cm.Spectral(np.linspace(0, 1, len(hdb_unique_labels)))
# fig = plt.figure(figsize=plt.figaspect(0.5))
# hdb_axis = fig.add_subplot('121')
#
# for k, col in zip(hdb_unique_labels, hdb_colors):
#     if k == -1:
#         # Black used for noise.
#         col = 'k'
#     ## Not sure what the x, y axes should be, this is how I had it set up for the previous run of HDBSCAN
#     ## Right now it's only comparing the first and second data points of all patients
#     hdb_axis.plot(preSpearman[hdb_labels == k, 0].astype(float), preSpearman[hdb_labels == k, 1].astype(float), 'o', markerfacecolor=col,
#                   markeredgecolor='k', markersize=6)
# hdb_axis.set_title('HDBSCAN\nEstimated number of clusters: %d' % n_clusters_hdb_)
#plt.show()
