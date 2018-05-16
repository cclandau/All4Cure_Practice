# Trying to implement the preprocessing.py using methods

from getTreatments import getTreatments
from getVectors import getVectors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import datetime
from datetime import datetime, date, time, timedelta
import math
from scipy import stats
from scipy.spatial import distance
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import MinMaxScaler
import time
import inspect
import csv
from ipykernel.tests.test_serialize import point
from sympy.polys.partfrac import apart
from sympy.polys.polytools import intervals
from docutils.writers.docutils_xml import RawXmlError
import os
#from scipy.io.arff.tests.test_arffread import DataTest
np.set_printoptions(threshold=np.inf)

# # Global Variable, setting the number of data points we initially look at to 5
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

def extractRawInfo():
    with open("PatientList.csv", "r") as patList:
        reader = csv.reader(patList)
        patList = np.array(list(reader))
        patDict = {}
        for i in range(1, patList.shape[0]):
            # KEY = MM-# VALUE = whether they are Kappa or Lambda
            patDict[patList[i][0]] = patList[i][3]

    with open("raw_KappaFLC.csv", "r") as rawKappaData:
        reader = csv.reader(rawKappaData)
        rawKapFLC = np.array(list(reader))

    with open("raw_LambdaFLC.csv", "r") as rawLambdaData:
        reader = csv.reader(rawLambdaData)
        rawLamFLC = np.array(list(reader))

    rawKapFLC = np.delete(rawKapFLC, np.s_[0], axis = 0)
    rawLamFLC = np.delete(rawLamFLC, np.s_[0], axis = 0)
    rawKapFLC = np.delete(rawKapFLC, np.s_[3,4], axis = 1)
    rawLamFLC = np.delete(rawLamFLC, np.s_[3,4], axis = 1)

    #Loop through Kappa patient records, deleting any records of Lambda patients
    #At the same time, read the date into a list of datetime objects for better use
    temp1 = 0
    temp2 = 0
    rawKapDates = []
    rawLamDates = []

    while temp1 != rawKapFLC.shape[0]:
        if patDict.get(rawKapFLC[temp1][0]) != "Kappa":
            rawKapFLC = np.delete(rawKapFLC, np.s_[temp1], axis = 0)
        else:
            rawKapFLC[temp1][2] = rawKapFLC[temp1][2].replace(" 00:00:00", "")
            if len(rawKapFLC[temp1][2]) == 8:
                rawKapFLC[temp1][2] = "0" + rawKapFLC[temp1][2]
                rawKapFLC[temp1][2] = rawKapFLC[temp1][2][0:3] + "0" + rawKapFLC[temp1][2][3:]
            elif len(rawKapFLC[temp1][2]) == 9:
                if rawKapFLC[temp1][2][0:2] == "11" or rawKapFLC[temp1][2][0:2] == "12" or rawKapFLC[temp1][2][0:2] == "10":
                    rawKapFLC[temp1][2] = rawKapFLC[temp1][2][0:3] + "0" + rawKapFLC[temp1][2][3:]
                else:
                    rawKapFLC[temp1][2] = "0" + rawKapFLC[temp1][2]
            rawKapDates.append(datetime.strptime(rawKapFLC[temp1][2], '%Y-%m-%d').date())
            temp1 = temp1 + 1

    #Do the same thing for the Lambda patient records
    while temp2 != rawLamFLC.shape[0]:
        if patDict.get(rawLamFLC[temp2][0]) != "Lambda":
            rawLamFLC = np.delete(rawLamFLC, np.s_[temp2], axis = 0)
        else:
            rawLamFLC[temp2][2] = rawLamFLC[temp2][2].replace(" 00:00:00", "")
            rawLamDates.append(datetime.strptime(rawLamFLC[temp2][2], '%Y-%m-%d').date())
            temp2 = temp2 + 1

    rawKapDates = np.array(rawKapDates)
    rawLamDates = np.array(rawLamDates)



    global raw_X
    raw_X = np.concatenate((rawKapFLC, rawLamFLC), axis=0)

    global raw_dates
    raw_dates = np.concatenate((rawKapDates, rawLamDates), axis=0)

    global dataTest
    dataTest = {}

    ## PATIENT, FLC VALUE, TEST DATE
    ## N X M MATRIX = ROWS X COLUMNS
    for i in range(0, raw_X.shape[0]):
        # If this patient is already in the dictionary, add additional FLC test values
        if raw_X[i][0] in dataTest:
            dataTest[raw_X[i][0]].append([raw_X[i][1], raw_dates[i]])
        else: # add to dictionary
            dataTest[raw_X[i][0]] = []
            dataTest[raw_X[i][0]].append([raw_X[i][1], raw_dates[i]])

    with open("rawValuesMatrix.csv", "w") as csvfile:
        for key,value in dataTest.items():
            csvfile.write(key + ': ')
            for v in value:
                csvfile.write(v[0] + ", ")
                csvfile.write(str(v[1]) + " ")
            csvfile.write('\n')

    with open("treatmentDictionary.csv", "w") as csvfile:
        for key,value in treatDict.items():
            csvfile.write(key + ': ')
            for v in value:
                csvfile.write(v[0] + ", ")
                csvfile.write(str(v[1]) + " ")
            csvfile.write('\n')

def rawDelete():
    for i in dataTest.keys():
        temp = dataTest[i]
        dataTest[i] = sorted(temp, key=lambda temp_entry: temp_entry[1])

    keysToDelete = []
    smolderingRawPatientsDict = {}

    for i in dataTest.keys():
        tempFLC = dataTest[i]
        if((tempFLC[np.array(dataTest[i]).shape[0] - 1][1] - tempFLC[0][1]).days <= 180):
            # print("patient with less than six months: " + i)
            keysToDelete.append(i)
        else:
            if i not in treatDict.keys():
                smolderingRawPatientsDict[i] = dataTest[i]
                keysToDelete.append(i)
                # print("smoldering patient: " + i)
            else:
                tempTreat = treatDict[i]
                if(tempFLC[0][1] > tempTreat[0][1]):  # #tempFLC[row][column] -> FLC DATE KAPPA/LAMBDA*
                    keysToDelete.append(i)
                    # print("patient with treatment before reading: " + i)
                haveFoundSixMonth = False
                for j in range(0, np.array(dataTest[i]).shape[0]):  # for every row in matrix
                    if (((tempFLC[j][1] - tempFLC[0][1]).days >= 180) and (haveFoundSixMonth != True)):
                        sixMonthIndex = j
                        haveFoundSixMonth = True
                firstSixMonths = np.array(tempFLC)[:(sixMonthIndex), :]
                #dataTest[i] = firstSixMonths
                #tempFLC = dataTest[i]
                # print("patient: " + i)
                # else:
                    # print("good patient: " + i)
    for i in keysToDelete:
        del dataTest[i]

    with open("rawValuesFilter_1.csv", "w") as csvfile:
        for key,value in dataTest.items():
            csvfile.write(key + ': ')
            for v in value:
                csvfile.write(v[0] + ", ")
                csvfile.write(str(v[1]) + " ")
            csvfile.write('\n')

def rawBinMaker():
    ## MAP
    ## KEY = MM-
    ## VALUE = DICTIONARY => KEY = NUMBER OF WEEKS - (3->5)*N
    ##                       VALUE = LIST OF FLC VALUES IN THAT TIME FRAME
    global outerDict
    global segmentLength
    global overlapBy
    segmentLength = 6
    #Change to desired overlap
    overlapBy = 0
    global medSequenceFileName
    global labSequenceFileName
    #Created medSequenceMatrix and labSequenceMatrix which is the csv files in matrix form
    #and without dates. We may not need labSequenceMatrix and we can delete it later if unused.
    global medSequenceMatrix
    global labSequenceMatrix
    medSequenceMatrix = []
    labSequenceMatrix = []
    medSequenceFileName = 'miniSeqsMedsSL_6_OL_0.csv'
    labSequenceFileName = 'miniSeqsLabsSL_6_OL_0.csv'
    try:
        os.remove(medSequenceFileName)
    except OSError:
        pass
    try:
        os.remove(labSequenceFileName)
    except OSError:
        pass
    outerDict = {}
    weekCounter = 1
    for eachKey, value in dataTest.items():
        allTests = []
        allDates = []
        #print("About to make inner dictionary for: " + eachKey)
        index = 0;
        for v in value:
            if (str(v[1])) in allDates:
                #print("replacing FLC VALUE " + allTests[index - 1] + " on " + allDates[index - 1] + " with " + v[0] + " from " + str(v[1]) + " for patient " + eachKey)
                allTests[index - 1] = v[0]
                allDates[index - 1] = str(v[1])
            else:
                allTests.append(v[0])
                allDates.append(str(v[1]))
                index = index + 1
        outerDict[eachKey] = properSampleMaker(eachKey, allTests, allDates)
    #print("Went through all patients, printing outer Dictionary NOW")
    with open("rawValuesBins_1.csv", "w", newline='') as csvfile:
        for x in outerDict:
            #print(x)
            # X is the Patient ID number
            csvfile.write(x)
            csvfile.write("\n")
            for y in outerDict[x]:
                treatment = str(outerDict[x][y][2]).replace(",", ";")
                temp = str(outerDict[x][y][0]) + ", " +  str(outerDict[x][y][1]) + ", " + treatment
                temp = temp.strip("'[]")
                csvfile.write("Bin: " + str(y) + ", " + temp.replace("'", ""))
                csvfile.write("\n")
                #print(y, ' : ' , outerDict[x][y])
    labSequenceMatrix = np.array(labSequenceMatrix)
    medSequenceMatrix = np.array(medSequenceMatrix)

def properSampleMaker(patientID, FLC_Value, Date):
    dataDict = {'Date': Date, 'FLC_Value': FLC_Value}
    df = pd.DataFrame(dataDict, columns = ['Date', 'FLC_Value'])
    df = df.set_index(pd.DatetimeIndex(df['Date']))
    del df['Date']
    df['FLC_Value'] = df['FLC_Value'].apply(pd.to_numeric, errors='coerce')
    resample = df.resample('D').mean()
    interpolated = resample.interpolate(method='linear')
    downSample = interpolated.resample('28D').first()
    finalData = downSample.reset_index()
    finalData = finalData.values
    innerDict = {}
    datesForTreat = []
    if (patientID == 'MM-120'):
#         with open('MM120_test.csv', 'w', newline='') as csvfile:
#             csvfile.write('\n'.join('{}, {}, {}'.format(resample[x], interpolated[x], downSample[x]) for x in range(0, len(interpolated.values))))
        #print("Original")
        #print(df)
        #print("Daily Upsample")
        #print(resample)
        #print("Interpolating Daily")
        #print(interpolated)
        #print("Down Sample 28 Day Interpolation")
        #print(downSample)
        pass

    for x in range(len(finalData) - 1):
        #treatmentArray = treatmentAdder(finalData[:, 0], np.array(treatDict[patientID]), len(finalData) - 1, patientID)
        innerDict[x] = [finalData[x, 0].to_pydatetime().strftime('%Y-%m-%d'), round(finalData[x, 1], 2)] # Date, FLC
        datesForTreat.append(finalData[x, 0].to_pydatetime().strftime('%Y-%m-%d'))
    treatMatrix = tryTreatment(datesForTreat, patientID)
    indexT = 0
    counterT = 0
    with open(medSequenceFileName, 'a') as csvfile:    #change to file name
        while(indexT + segmentLength < len(datesForTreat)):
            temp = []
            csvfile.write(str(patientID) + "." + str(counterT))
            temp.append(str(patientID) + "." + str(counterT))
            csvfile.write(",")
            for i in range(indexT, segmentLength + indexT):
                csvfile.write(str(treatMatrix[i]).replace(',', ';') + "," + datesForTreat[i])
                temp.append(treatMatrix[i])
                csvfile.write(',')
            csvfile.write('\n')
            medSequenceMatrix.append(temp)
            indexT = indexT + segmentLength - overlapBy
            counterT += 1
    indexL = 0
    counterL = 0
    with open(labSequenceFileName, 'a') as csvfile:  #change to file name
        while(indexL + segmentLength < len(datesForTreat)):
            temp = []
            csvfile.write(str(patientID) + "." + str(counterL))
            temp.append(str(patientID) + "." + str(counterL))
            csvfile.write(",")
            for i in range(indexL, segmentLength + indexL):
                tempList = innerDict[i]
                csvfile.write(str(tempList[1]) + "," + datesForTreat[i])
                temp.append(tempList[1])
                csvfile.write(',')
            csvfile.write('\n')
            labSequenceMatrix.append(temp)
            indexL = indexL + segmentLength - overlapBy
            counterL += 1
    for x in range(len(finalData) - 1):
        innerDict[x].append(treatMatrix[x])
    #with open('')
    #print(innerDict)
    return innerDict

def getDistancesFromMeds():
    treatmentData = np.delete(medSequenceMatrix, 0, 1)
    #print(treatmentData)
    noBits = 24
    signBitOption = 1 #Set to 1 if you want to use the sign bit (AKA if you
    # want to keep track of unknown drugs.)
    monthlyBinaryTreatmentVectors = []
    for i in range(0, treatmentData.shape[0]):
        temp = []
        for j in range(0, segmentLength):
            binaryTreatment = convTreatmentToBinaryArray(treatmentData[i, j], signBitOption, noBits)
            temp.append(binaryTreatment)
        monthlyBinaryTreatmentVectors.append(temp)
    monthlyBinaryTreatmentVectors = np.array(monthlyBinaryTreatmentVectors)
    aggregateTreatmentsOption = 0 #Set to 0 for keeping monthly granularity. Set
    # to 1 to encode all 6 months as one aggregate binary string
    #print(monthlyBinaryTreatmentVectors.astype(bool))
    binaryTreatmentVectors = []
    if(aggregateTreatmentsOption):
        boolMonthlyTreatVectors = monthlyBinaryTreatmentVectors.astype(bool)
        for i in range(monthlyBinaryTreatmentVectors.shape[0]):
            rowOfMonths = np.zeros(noBits + signBitOption).astype(bool)
            for j in range(monthlyBinaryTreatmentVectors.shape[1]):
                rowOfMonths = np.logical_or(rowOfMonths, boolMonthlyTreatVectors[i, j])
            binaryTreatmentVectors.append(rowOfMonths.astype(int))
    else:
        for i in range(monthlyBinaryTreatmentVectors.shape[0]):
            rowOfMonths = np.array([])
            for j in range(monthlyBinaryTreatmentVectors.shape[1]):
                rowOfMonths = np.concatenate([rowOfMonths, monthlyBinaryTreatmentVectors[i, j]])
            binaryTreatmentVectors.append(rowOfMonths)
    binaryTreatmentVectors = np.array(binaryTreatmentVectors)
    #print(binaryTreatmentVectors)
    # Set binaryDistanceMethod = 1 for Sokal Michener
    # Set binaryDistanceMethod = 2 for Jaccard
    # Set binaryDistanceMethod = 3 for Rogers Tanimoto
    # Set binaryDistanceMethod = 4 for Sokal Sneath II
    binaryDistanceMethod = 2
    distanceMatrix = np.zeros((len(binaryTreatmentVectors), len(binaryTreatmentVectors)))
    # Set useManualDistanceMethod = 1 to compute distances manually, set useManualDistanceMethod = 0
    # to use the SciPy distance calculations
    useManualDistanceMethod = 1
    if binaryDistanceMethod == 1:
        for i in range(binaryTreatmentVectors.shape[0]):
            for j in range(i, binaryTreatmentVectors.shape[0]):
                if useManualDistanceMethod:
                    dist = np.sum(np.logical_xor(binaryTreatmentVectors[i].astype(bool), binaryTreatmentVectors[j].astype(bool)))/(noBits + signBitOption)
                else:
                    dist = distance.hamming(binaryTreatmentVectors[i].astype(bool), binaryTreatmentVectors[j].astype(bool))
                distanceMatrix[i, j] = dist
                distanceMatrix[j, i] = dist
    elif binaryDistanceMethod == 2:
        for i in range(binaryTreatmentVectors.shape[0]):
            for j in range(i, binaryTreatmentVectors.shape[0]):
                if np.array_equal(binaryTreatmentVectors[i], binaryTreatmentVectors[j]):
                    dist = 0
                else:
                    if useManualDistanceMethod:
                        numerator = np.sum(np.logical_xor(binaryTreatmentVectors[i].astype(bool), binaryTreatmentVectors[j].astype(bool)))
                        denomenator = np.sum(np.logical_or(binaryTreatmentVectors[i].astype(bool), binaryTreatmentVectors[j].astype(bool)))
                        dist = numerator/denomenator
                    else:
                        dist = distance.jaccard(binaryTreatmentVectors[i].astype(bool), binaryTreatmentVectors[j].astype(bool))
                distanceMatrix[i, j] = dist
                distanceMatrix[j, i] = dist
    elif binaryDistanceMethod == 3:
        for i in range(binaryTreatmentVectors.shape[0]):
            for j in range(i, binaryTreatmentVectors.shape[0]):
                if useManualDistanceMethod:
                    numerator = 2 * np.sum(np.logical_xor(binaryTreatmentVectors[i].astype(bool), binaryTreatmentVectors[j].astype(bool)))
                    denomenator = np.sum(np.logical_xor(binaryTreatmentVectors[i].astype(bool), binaryTreatmentVectors[j].astype(bool))) + noBits + signBitOption
                    dist = numerator/denomenator
                else:
                    dist = distance.rogerstanimoto(binaryTreatmentVectors[i].astype(bool), binaryTreatmentVectors[j].astype(bool))
                distanceMatrix[i, j] = dist
                distanceMatrix[j, i] = dist
    else:
        for i in range(binaryTreatmentVectors.shape[0]):
            for j in range(binaryTreatmentVectors.shape[0]):
                if np.array_equal(binaryTreatmentVectors[i], binaryTreatmentVectors[j]):
                    dist = 0
                else:
                    if useManualDistanceMethod:
                        numerator = 2 * np.sum(np.logical_xor(binaryTreatmentVectors[i].astype(bool), binaryTreatmentVectors[j].astype(bool)))
                        denomenator = (2 * np.sum(np.logical_xor(binaryTreatmentVectors[i].astype(bool), binaryTreatmentVectors[j].astype(bool)))) + np.sum(np.logical_and(binaryTreatmentVectors[i].astype(bool), binaryTreatmentVectors[j].astype(bool)))
                        dist = numerator/denomenator
                    else:
                        dist = distance.sokalsneath(binaryTreatmentVectors[i].astype(bool), binaryTreatmentVectors[j].astype(bool))
                distanceMatrix[i, j] = dist
    adjustScaleOfDistanceMatrix = 1
    if adjustScaleOfDistanceMatrix:
        max = np.amax(distanceMatrix)
        distanceMatrix = np.divide(distanceMatrix, max)
    np.savetxt('distanceMatrix.csv', distanceMatrix, delimiter = ',')
    #ex1 = np.array([1, 0, 1, 0, 0]).astype(bool)
    #ex2 = np.array([1, 0, 0, 0, 1]).astype(bool)
    #print(distance.sokalsneath(ex1, ex2))
    #print(distanceMatrix)
    return distanceMatrix

def clusterFromDistMatrix(distanceMatrix):
    clusterer = HDBSCAN(min_cluster_size = 2, metric = 'precomputed')
    clusterer.fit(distanceMatrix)
    labels = clusterer.labels_
    probs = clusterer.probabilities_
    labels = np.array((labels))[np.newaxis]
    labels = labels.T
    probs = np.array((probs))[np.newaxis]
    probs = probs.T
    results = np.concatenate((probs, medSequenceMatrix[0:50, :]), axis = 1)
    results = np.concatenate((labels, results), axis = 1)
    results = np.array(sorted(results, key=lambda a_entry: a_entry[0]))
    #print(results)
    with open('treatmentClusters.txt', 'w') as csvfile:
        csvfile.write("Cluster; Probability; ID; Month 1; Month 2; Month 3; Month 4; Month 5; Month 6")
        csvfile.write('\n')
        for i in range(results.shape[0]):
            csvfile.write(str(results[i, 0]))
            csvfile.write(';')
            csvfile.write(str(results[i, 1]))
            csvfile.write(';')
            csvfile.write(str(results[i, 2]))
            csvfile.write(';')
            for j in range(3, results.shape[1]):
                csvfile.write(str(results[i, j]).replace('{', '').replace('}', ''))
                csvfile.write(';')
            csvfile.write('\n')



    #for i in range(50):
    #    print(medSequenceMatrix[i, :])
    #print(labels)
    #print(probs)

def convTreatmentToBinaryArray(treatment, signBitOption, noBits):
    signBit = 0
    if -1 in treatment:
        signBit = 1
    binaryTreatment = np.zeros(noBits)
    if 0 not in treatment:
        listTreatments = list(treatment)
        for i in range(len(listTreatments)):
            if listTreatments[i] != -1:
                binaryTreatment[listTreatments[i] - 1] = 1
    if signBitOption:
        binaryTreatment = np.insert(binaryTreatment, 0, signBit, axis=0)
    return binaryTreatment

def patientBinCreator(patientID, FLC_Value, Date):
    #innerDict = {'0' : ['test initial'], '1' : ['a', 'b', 'c']}
    for x in range(0, len(Date)):
        Date[x] = datetime.strptime(Date[x], '%Y-%m-%d').date()
    innerDict = {}
    innerDict[0] = [FLC_Value[0], Date[0]]
    binNumber = 0
    dateToCompare = Date[0]
    for dateInstance in range(1, len(Date) - 1):
        timeDiff = Date[dateInstance] - dateToCompare
        timeDifference = timeDiff.days
        # If it's within 21 to 35 days from the previous date marker, put in current bin
        if (timeDifference > 21 and timeDifference < 35):
            if (binNumber == 0):
                binNumber = 1
            if (binNumber in innerDict):
                innerDict[binNumber].extend([FLC_Value[dateInstance], Date[dateInstance]])
            else:
                innerDict[binNumber] = [FLC_Value[dateInstance], Date[dateInstance]]
        # Time difference between current point and date to compare falls outside of the bin range,
        # increase the bin number by one, and later fill that bin with an interpolated (linear) value
        elif (timeDifference > 35):
            binNumber = binNumber + 1
            #print("Not within range, starting bin number " + str(binNumber) + "!")
            treatmentsArray = treatDict[patientID]
            currentTreat = []
            oldTreat = []
            for x in range(0, len(treatmentsArray) - 1):
                # Wait to actually update dateToCompare so we can easily fill oldTreat array
                dateInterest = treatmentsArray[x][1] - dateToCompare + timedelta(days=28)
                dateInt = dateInterest.days
                if (dateInt > 21 and dateInt < 35):
                    if (treatmentsArray[x][0] not in currentTreat):
                        currentTreat.append(treatmentsArray[x][0])
                if (timeDifference > 21 and timeDifference < 35):
                    if (treatmentsArray[x][0] not in oldTreat):
                        oldTreat.append(treatmentsArray[x][0])
            dateToCompare = dateToCompare + timedelta(days=28)
            if (binNumber in innerDict):
                innerDict[binNumber].extend(["-1", dateToCompare])
            else:
                innerDict[binNumber] = ["-1", dateToCompare]
        else:
#             print("Too small for range, delete this item")
#             print(str(timeDifference))
            pass
    print(innerDict)
    print("About to return to rawBinMaker")
    return innerDict


def derivativeMaker():
    # Create first derivative column
    D1 = np.zeros((len(X), 1))
    for i in range(1, len(dates) - 1):
        if X[i][0] == X[i - 1][0]:
            xdif = dates[i] - dates[i - 1]
            ydif = np.float(X[i][1]) - np.float(X[i - 1][1])
            if ydif == 0:
                D1[i] = 0
            elif xdif.total_seconds() / 86400 == 0:
                if ydif > 0:
                    D1[i] = float('inf')
                else:
                    D1[i] = float('-inf')
            else:
                D1[i] = ydif / (xdif.total_seconds() / 86400)
    min_val = D1[np.isfinite(D1)].min()
    max_val = D1[np.isfinite(D1)].max()
    for i in range(1, len(dates) - 1):
        if D1[i] == float('-inf'):
            D1[i] = min_val
        elif D1[i] == float('inf'):
            D1[i] = max_val

    # Create second derivative column
    D2 = np.zeros((len(X), 1))
    for i in range(1, len(dates) - 1):
        if X[i][0] == X[i - 1][0]:
            xdif = dates[i] - dates[i - 1]
            ydif = np.float(D1[i]) - np.float(D1[i - 1])
            if ydif == 0:
                D2[i] = 0
            elif xdif.total_seconds() / 86400 == 0:
                if ydif > 0:
                    D2[i] = float('inf')
                else:
                    D2[i] = float('-inf')
            else:
                D2[i] = ydif / (xdif.total_seconds() / 86400)
    min_val2 = D2[np.isfinite(D2)].min()
    max_val2 = D2[np.isfinite(D2)].max()
    for i in range(1, len(dates) - 1):
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
            # print("patient with less than six months: " + i)
            keysToDelete.append(i)
        else:
            if i not in treatDict.keys():
                smolderingPatientsDict[i] = FLCdict[i]
                keysToDelete.append(i)
                # print("smoldering patient: " + i)
            else:
                tempTreat = treatDict[i]
                if(tempFLC[0][1] > tempTreat[0][1]):  # #tempFLC[row][column] -> FLC DATE KAPPA/LAMBDA*
                    keysToDelete.append(i)
                    # print("patient with treatment before reading: " + i)
                haveFoundSixMonth = False
                for j in range(0, np.array(FLCdict[i]).shape[0]):  # for every row in matrix
                    if (((tempFLC[j][1] - tempFLC[0][1]).days >= 180) and (haveFoundSixMonth != True)):
                        sixMonthIndex = j
                        haveFoundSixMonth = True
                firstSixMonths = np.array(tempFLC)[:(sixMonthIndex), :]
                FLCdict[i] = firstSixMonths
                tempFLC = FLCdict[i]
                # print("patient: " + i)
                # else:
                    # print("good patient: " + i)
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
    lengthSegment = 5  # # Should we leave this as different than numberOfPoints?
    with open('processed.csv', 'w') as csvfile:
        temp = []
        for i in FLCdict.keys():
            temp = np.array(FLCdict[i])
            numReadings = temp.shape[0]  # ##numReadings = number of readings left to process
            counter = 0
            while(numReadings >= lengthSegment):  # will need to change 5 to a field
                temp2 = []
                for j in range((lengthSegment - 1) * counter, lengthSegment + (lengthSegment - 1) * counter):
                    temp2.append(temp[j][0])
                csvfile.write(str(i) + "-" + str(counter))
                useablePatients.append(str(i) + "-" + str(counter))
                for j in range (((lengthSegment - 1) * counter), lengthSegment + ((lengthSegment - 1) * counter)):
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
    # print("FLC Value of each element, Index, Row, minValue and maxValue")
    for patient in np.nditer(np.array(preSpearman.astype(float))):
        # print("'{0}', '{1}', '{2}', '{3}'".format(patient, index, row, minValues[row]))
        temp = np.array(patient)
        temp = temp - minValues[row]  # will need to change to be a field
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


# ## parameter for hdbProcessing should be:
    # unprocessed - np.array(preSpearman.astype(float))
    # minMax - minMaxMatrix
    # logScaled - logRawMatrix

def initialHDBProcessing(workingMatrix, selector):
    hdb_t1 = time.time()
    #print(workingMatrix[:,1:].astype(float))
    hdb = HDBSCAN(min_cluster_size=2).fit(np.delete(workingMatrix[:,1:].astype(float), 0, axis=1))
    hdb_labels = hdb.labels_
    hdb_prob = hdb.probabilities_
    hdb_elapsed_time = time.time() - hdb_t1
    n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    merged = np.array(list(zip(workingMatrix[:,0], hdb_labels, hdb_prob)))

    with open(selector + '.csv', 'w') as csvfile:
        csvfile.write("Patient Number, Cluster Label, Cluster Probability" + '\n')
        csvfile.write('\n'.join('{}, {}, {}'.format(x[0], x[1], x[2]) for x in merged))

    #print('\n\nHDBSCAN Results for ' + selector)
    #print('Estimated number of clusters: %d' % n_clusters_hdb_)
    #print('Patients, Cluster Labels, and Probability')
    return(merged)

    ## hdb ___ plotting ###
    #hdb_unique_labels = set(hdb_labels)
    #hdb_colors = plt.cm.Spectral(np.linspace(0, 1, len(hdb_unique_labels)))
    #fig = plt.figure(figsize=plt.figaspect(0.5))
    #hdb_axis = fig.add_subplot('121')

    #for k, col in zip(hdb_unique_labels, hdb_colors):
    #    if k == -1:
            # Black used for noise.
    #        col = 'k'
        # # Not sure what the x, y axes should be, this is how I had it set up for the previous run of HDBSCAN
        # # Right now it's only comparing the first and second data points of all patients
    #    hdb_axis.plot(workingMatrix[hdb_labels == k, 0].astype(float), workingMatrix[hdb_labels == k, 1].astype(float), 'o', markerfacecolor=col,
    #                  markeredgecolor='k', markersize=6)
    #hdb_axis.set_title('HDBSCAN\nEstimated number of clusters: %d' % n_clusters_hdb_)
    # plt.show()


# this function takes in either a spearman or pearson correlation matrix and
# outputs the proper distance matrix

def distanceMatrix(correlationMatrix):
    calculatedMatrix = np.add(correlationMatrix, 1)  # shifts the correlation matrix by 1
    calculatedMatrix = np.divide(calculatedMatrix, -2)  # divides the entire matrix by -2
    calculatedMatrix = np.add(calculatedMatrix, 1)  # subtracts each value from 1
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
    np.savetxt("pearson.csv", pearson, delimiter=",")
    return pearson

def tauCorr():
    tau = np.zeros((preSpearman.shape[0], preSpearman.shape[0]))
    for i in range(0, preSpearman.shape[0]):
        for j in range(0, preSpearman.shape[0]):
            calculatedTau = stats.kendalltau(preSpearman[i, :].astype(float), preSpearman[j, :].astype(float))
            corr = calculatedTau[0]
            tau[i, j] = corr
            tau[j, i] = corr
    calculatedTau = np.round(calculatedTau, 1)
    np.savetxt("tau.csv", tau, delimiter=",")
    return tau

def tryTreatment(Dates, Patient):
    #print
    # (Dates)
    binDates = []
    treatments = []
    temp = {}
    for i in range(0, len(Dates)):
        binDates.append(datetime.strptime(Dates[i], '%Y-%m-%d').date())
    patientsArray = np.array(treatDict[Patient])
    numData = patientsArray.shape[0]
    temp[0] = set()
    temp[0].add(0)
    for i in range(0, numData):
        treatment = patientsArray[i, 0]
        currDate = patientsArray[i, 1]
        for j in range(1, len(binDates)):
            if currDate >= binDates[j-1] and currDate < binDates[j]:
                if j not in temp.keys():
                    temp[j] = set()
                if(treatment == 'Lenalidomide'):
                    temp[j].add(1)
                elif(treatment == 'Bortezomib'):
                    temp[j].add(2)
                elif(treatment == 'Carfilzomib'):
                    temp[j].add(3)
                elif(treatment == 'Dexamethasone'):
                    temp[j].add(4)
                elif(treatment == 'Pomalidomide'):
                    temp[j].add(5)
                elif(treatment == 'Thalidomide'):
                    temp[j].add(6)
                elif(treatment == 'Cyclophosphamide'):
                    temp[j].add(7)
                elif(treatment == 'Melphalan'):
                    temp[j].add(8)
                elif(treatment == 'Prednisone'):
                    temp[j].add(9)
                elif(treatment == 'Ixazomib'):
                    temp[j].add(10)
                elif(treatment == 'Cisplatin'):
                    temp[j].add(11)
                elif(treatment == 'Doxorubicin'):
                    temp[j].add(12)
                elif(treatment == 'Etoposide'):
                    temp[j].add(13)
                elif(treatment == 'Vincristine'):
                    temp[j].add(14)
                elif(treatment == 'Daratumumab'):
                    temp[j].add(15)
                elif(treatment == 'Elotuzumab'):
                    temp[j].add(16)
                elif(treatment == 'Bendamustine'):
                    temp[j].add(17)
                elif(treatment == 'Panobinostat'):
                    temp[j].add(18)
                elif(treatment == 'Venetoclax'):
                    temp[j].add(19)
                elif(treatment == 'CAR-T'):
                    temp[j].add(20)
                else:
                    temp[j].add(-1)
    for k in range(0, len(binDates)):
        if k not in temp.keys():
            temp[k] = set()
            temp[k].add(0)
    for i in range(0, len(binDates)):
        treatments.append(temp[i])
    return treatments

#Will build a matrix from the raw and interpolated values in
#the miniSeqsLabsSL_6_OL_1.csv file. This raw data matrix will then
#be used in our clustering algorithms.
def buildMatrix():
    with open("miniSeqsLabsSL_6_OL_1.csv", "r") as rawPatientData:
        reader = csv.reader(rawPatientData)
        binnedData = np.empty(numberOfPoints + 2)
        for line in list(reader):
            column = 1
            justData = [] #this matrix will have patient number and each data point
            justData.append(line[0])
            while column < len(line) - 1:
                justData.append(float(line[column].split(":", 1)[0])) #gets rid of the date attached to each FLC value
                column = column + 1
            binnedData = np.vstack((binnedData, justData))
        binnedData = np.delete(binnedData, (0), axis=0)
        return binnedData

#Will min-max scale the data in binnedData, and then run hdbscan on the new matrix.
#runs a min-max normalization on the
def getClustersOnTrend(toBeNormalized):
    for i in range(0, toBeNormalized.shape[0]):
       minFLC = min(toBeNormalized[i][1:].astype(float))
       for j in range(1, len(toBeNormalized[i])):
           toBeNormalized[i][j] = (toBeNormalized[i][j].astype(float) - minFLC.astype(float))
       maxFLC = max(toBeNormalized[i][1:].astype(float))
       for j in range(1, len(toBeNormalized[i])):
           if(maxFLC.astype(float) != 0):
               toBeNormalized[i][j] = (toBeNormalized[i][j].astype(float))/maxFLC.astype(float)
    clusters = initialHDBProcessing(toBeNormalized, "hdbscanBinnedData")
    sortedClusters = clusters[clusters[:,1].astype(float).argsort()]
    return sortedClusters

def partitionTrendClustersOnScale(rawData, clusters):
    currentCluster = -1
    clusterMatrix = np.empty(numberOfPoints + 2)
    for row in clusters:
        if(int(row[1]) != currentCluster):
            clusterMatrix = np.delete(clusterMatrix, 0, axis=0)
            print(clusterMatrix.shape[0])
            print("array for cluster: ", currentCluster)
            print(clusterMatrix, '\n\n\n\n')
            #for clusterRow in range(0, clusterMatrix.shape[0]): #plotting
            #     plt.plot([1,2,3,4,5,6], clusterMatrix[clusterRow,1:].astype(float), label=clusterMatrix[clusterRow, 0])
            #     plt.legend()
            #plt.title("cluster : " + str(currentCluster))
            #plt.show()
            #input()
            #plt.clf()
            subClusters = initialHDBProcessing(clusterMatrix, "subClusterData" + str(currentCluster))
            currentCluster = int(row[1])
            clusterMatrix = np.empty(numberOfPoints + 2)
        for patient in rawData:
            if(patient[0] == row[0]):
                clusterMatrix = np.vstack((clusterMatrix, patient))
                break
    clusterMatrix = np.delete(clusterMatrix, 0, axis=0)



extractInfo()
extractRawInfo()
rawDelete()
rawBinMaker()
treatmentDistances = getDistancesFromMeds()
clusterFromDistMatrix(treatmentDistances[0:50, 0:50])
#binnedData = buildMatrix()
#rawData = np.copy(binnedData)
#sortedClusters = getClustersOnTrend(binnedData)
#partitionTrendClustersOnScale(rawData, sortedClusters)
#print(binnedData)
#D1, D2 = derivativeMaker()
#FLCdictionary(D1, D2)
#processingWrite()
#treatCombDict()
# spearmanMatrix = spearmansCorr()
# spearmanDistanceMatrix = distanceMatrix(spearmanMatrix)
# pearsonsMatrix = pearsonsCorr()
# pearsonsDistanceMatrix = distanceMatrix(pearsonsMatrix)
#minMaxMatrix = minMaxNormalization()
#print(minMaxMatrix)
# logRawMatrix = logScaling()
# tauMatrix = tauCorr()
# tauDistanceMatrix = distanceMatrix(tauMatrix)
# hdbProcessing(np.array(preSpearman.astype(float)), "hdbscanPairs_unscaled")
# hdbProcessing(minMaxMatrix, "hdbscanPairs_minmax")
# hdbProcessing(logRawMatrix, "hdbscanPairs_log10")
# hdbProcessing(pearsonsDistanceMatrix, "hdbscanPairs_pearsons")
# hdbProcessing(spearmanDistanceMatrix, "hdbscanPairs_spearman")
# hdbProcessing(tauDistanceMatrix, "hdbscanPairs_tau")
#
# np.savetxt("pearsonDistance.csv", pearsonsDistanceMatrix, delimiter=",")
# np.savetxt("spearmanDistance.csv", spearmanDistanceMatrix, delimiter=",")
# np.savetxt("tauDistance.csv", tauDistanceMatrix, delimiter=",")
#
#this part is just for testing the accuracy of pearson and spearman clustering
#for row in range(0, rawData.shape[0]):
#     plt.clf()
#     plt.plot([1,2,3,4,5,6], rawData[row,1:].astype(float))
#     patNumber = rawData[row][0]
#     plt.title(patNumber)
#     plt.savefig("Graphs/" + patNumber + ".png")
