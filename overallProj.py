# Trying to implement the preprocessing.py using methods

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
from sklearn.datasets.samples_generator import make_blobs
import time
import inspect
import csv
np.set_printoptions(threshold=np.nan)

## Global Variable, setting the number of data points we initially look at to 5
numberOfPoints = 5

def extractInfo():
    unpack = getVectors()
    kapFLC = unpack[0]
    kapDates = unpack[2]
    lamFLC = unpack[3]
    lamDates = unpack[5]
    global X
    X = np.concatenate((kapFLC, lamFLC), axis=0)
    global dates
    dates = np.concatenate((kapDates, lamDates), axis=0)
    global treatDict
    treatDict = getTreatments()
    return

def derivativeMaker():
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

    return D1, D2

def FLCdictionary(D1, D2):
    global FLCdict
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
                #else:
                    #print("good patient: " + i)
    for i in keysToDelete:
        del FLCdict[i]
    ### plotting flc value for each patient ###
    # for i in FLCdict.keys():
    #    tempFLC = np.array(FLCdict[i])
    #    plt.figure()
    #    plt.plot(tempFLC[:, 1], tempFLC[:, 0])
    #    plt.title(i)
    #    # print(tempFLC[:, 0])
    #    # print(tempFLC[:, 1])
    #    plt.show()
    return

def processingWrite():
    global preSpearman
    preSpearman = []
    global useablePatients
    useablePatients = []
    lengthSegment = 5
    with open('processed.csv', 'w') as csvfile:
        temp = []
        for i in FLCdict.keys():
            temp = np.array(FLCdict[i])
            numReadings = temp.shape[0] ###numReadings = number of readings left to process
            counter = 0
            while(numReadings >= lengthSegment): # will need to change 5 to a field
                temp2 = []
                for j in range((lengthSegment-1)*counter, lengthSegment + (lengthSegment-1)*counter):
                    temp2.append(temp[j][0])
                csvfile.write(str(i) + "-" + str(counter))
                useablePatients.append(str(i) + "-" + str(counter))
                for j in range (((lengthSegment - 1)*counter), lengthSegment + ((lengthSegment - 1)*counter)):
                    csvfile.write(", " + str(temp[j][0]))
                    csvfile.write(", " + str(temp[j][1]))
                csvfile.write('\n')
                preSpearman.append(temp2)
                numReadings = numReadings - lengthSegment + 1
                counter += 1
    preSpearman = np.array(preSpearman)
    preSpearmanNum = np.array(preSpearman.astype(float))

    unprocessedMatrix = np.array(list(zip(useablePatients, np.array(preSpearman.astype(float)))), dtype=object)
    with open('unscaledData.csv', 'w') as csvfile:
        csvfile.write("Patient Number + UnScaled FLC Values" + '\n')
        csvfile.write('\n'.join('{}, {}'.format(x[0], x[1]) for x in unprocessedMatrix))
    return

def spearmansCorr():
    spearman = np.zeros((preSpearman.shape[0], preSpearman.shape[0]))
    for i in range(0, preSpearman.shape[0]):
        for j in range(0, preSpearman.shape[0]):
            SpearR = stats.spearmanr(preSpearman[i, :], preSpearman[j, :])
            corr = SpearR[0]
            spearman[i, j] = corr
            spearman[j, i] = corr
    spearman = np.round(spearman, 1)
    np.savetxt("spearman.csv", spearman, delimiter=",")
    return spearman

def minMaxNormalization():
    minValues = np.array(preSpearman.astype(float).min(axis=1))
    minMaxMatrix = np.zeros((len(minValues), numberOfPoints))
    index = 0;
    row = 0;
    #print("FLC Value of each element, Index, Row, minValue and maxValue")
    for patient in np.nditer(np.array(preSpearman.astype(float))):
        #print("'{0}', '{1}', '{2}', '{3}'".format(patient, index, row, minValues[row]))
        temp = np.array(patient)
        temp = temp - minValues[row] # will need to change to be a field
        # Subtracting original min portion
        minMaxMatrix[row, index] = temp
        index = index + 1
        # in the last column, need to determine max value of new row & update
        if (index == numberOfPoints):
            maxVal = minMaxMatrix[row, :].max()
            minMaxMatrix[row, :] = minMaxMatrix[row, :] / maxVal
            index = 0
            row = row + 1
    normalizedWithPatientNum = np.array(list(zip(useablePatients, minMaxMatrix)), dtype=object)

    with open('minMaxNormalized.csv', 'w') as csvfile:
        csvfile.write("Patient Number + Normalized FLC Values")
        csvfile.write('\n')
        csvfile.write('\n'.join('{}, {}'.format(x[0], x[1]) for x in normalizedWithPatientNum))
    return minMaxMatrix

def logScaling():
    logRawMatrix = np.log10(preSpearman.astype(float) + 1)
    logScaledWithPatientNum = np.array(list(zip(useablePatients, logRawMatrix)), dtype=object)

    with open('log10Scaled.csv', 'w') as csvfile:
        csvfile.write("Patient Number + Log Normalized FLC Values")
        csvfile.write('\n')
        csvfile.write('\n'.join('{}, {}'.format(x[0], x[1]) for x in logScaledWithPatientNum))
    return logRawMatrix

### parameter for hdbProcessing should be:
    # unprocessed - np.array(preSpearman.astype(float))
    # minMax - minMaxMatrix
    # logScaled - logRawMatrix
def hdbProcessing(workingMatrix, selector):
    hdb_t1 = time.time()
    hdb = HDBSCAN(min_cluster_size=2).fit(workingMatrix)
    hdb_labels = hdb.labels_
    hdb_prob = hdb.probabilities_
    hdb_elapsed_time = time.time() - hdb_t1
    n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    merged = np.array(list(zip(useablePatients, hdb_labels, hdb_prob)))


    with open(selector + '.csv', 'w') as csvfile:
        csvfile.write("Patient Number, Cluster Label, Cluster Probability" + '\n')
        csvfile.write('\n'.join('{}, {}, {}'.format(x[0], x[1], x[2]) for x in merged))


    print('\n\nHDBSCAN Results for ' + selector)
    print('Estimated number of clusters: %d' % n_clusters_hdb_)
    print('Patients, Cluster Labels, and Probability')
    print(merged)

    ## hdb ___ plotting ###
    hdb_unique_labels = set(hdb_labels)
    hdb_colors = plt.cm.Spectral(np.linspace(0, 1, len(hdb_unique_labels)))
    fig = plt.figure(figsize=plt.figaspect(0.5))
    hdb_axis = fig.add_subplot('121')

    for k, col in zip(hdb_unique_labels, hdb_colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
        ## Not sure what the x, y axes should be, this is how I had it set up for the previous run of HDBSCAN
        ## Right now it's only comparing the first and second data points of all patients
        hdb_axis.plot(workingMatrix[hdb_labels == k, 0].astype(float), workingMatrix[hdb_labels == k, 1].astype(float), 'o', markerfacecolor=col,
                      markeredgecolor='k', markersize=6)
    hdb_axis.set_title('HDBSCAN\nEstimated number of clusters: %d' % n_clusters_hdb_)
    #plt.show()
    return;

#this function takes in either a spearman or pearson correlation matrix and
#outputs the proper distance matrix
def distanceMatrix(correlationMatrix):
    calculatedMatrix = np.add(correlationMatrix, 1) #shifts the correlation matrix by 1
    calculatedMatrix = np.divide(calculatedMatrix, -2) #divides the entire matrix by -2
    calculatedMatrix = np.add(calculatedMatrix, 1) #subtracts each value from 1
    return calculatedMatrix

def pearsonsCorr():
    pearson = np.zeros((preSpearman.shape[0], preSpearman.shape[0]))
    for i in range(0, preSpearman.shape[0]):
        for j in range(0, preSpearman.shape[0]):
            calculatedPearson = stats.pearsonr(preSpearman[i, :].astype(float), preSpearman[j, :].astype(float))
            corr = calculatedPearson[0]
            pearson[i, j] = corr
            pearson[j, i] = corr
    calculatedPearson = np.round(calculatedPearson, 1)
    np.savetxt("pearson.csv",pearson, delimiter=",")
    return pearson



extractInfo()
D1, D2 = derivativeMaker()
FLCdictionary(D1, D2)
processingWrite()
spearmanMatrix = spearmansCorr()
spearmanDistanceMatrix = distanceMatrix(spearmanMatrix)
pearsonsMatrix = pearsonsCorr()
pearsonsDistanceMatrix = distanceMatrix(pearsonsMatrix)
minMaxMatrix = minMaxNormalization()
logRawMatrix = logScaling()
hdbProcessing(np.array(preSpearman.astype(float)), "hdbscanPairs_unscaled")
hdbProcessing(minMaxMatrix, "hdbscanPairs_minmax")
hdbProcessing(logRawMatrix, "hdbscanPairs_log10")
hdbProcessing(pearsonsDistanceMatrix, "hdbscanPairs_pearsons")
hdbProcessing(spearmanDistanceMatrix, "hdbscanPairs_spearman")

np.savetxt("pearsonDistance.csv", pearsonsDistanceMatrix, delimiter=",")
np.savetxt("spearmanDistance.csv", spearmanDistanceMatrix, delimiter=",")

#this part is just for testing the accuracy of pearson and spearman clustering
for row in range(0, preSpearman.shape[0]):
    plt.clf()
    plt.plot([1,2,3,4,5], preSpearman[row, :].astype(float))
    patNumber = open("processed.csv", "r")
    reader = csv.reader(patNumber)
    patList = np.array(list(reader))
    plt.title(patList[row, 0])
    plt.savefig(patList[row, 0])