## Cece's attempt at iterating through Lambda & Kappa FLC
## files to create a vector

import csv
import numpy as np
np.set_printoptions(threshold=np.nan)


def getKappaLambda():
    csvFile = open('kappaVALUES.csv', 'w')
    csvFileTwo = open('lambdaVALUES.csv', 'w')
    kappaFile = open('KappaFLC.csv', 'r')
    lambdaFile = open('LambdaFLC.csv', 'r')

    csvWrite = csv.writer(csvFile, quoting=csv.QUOTE_ALL)
    csvWriteTwo = csv.writer(csvFileTwo, quoting=csv.QUOTE_ALL)
    kappaRead = csv.reader(kappaFile)
    lambdaRead = csv.reader(lambdaFile)

    kappaMatrix = []
    lambdaMatrix = []

    # If we want all of the data from initial CSV files, simply replace
    # "matrix.append(reqData#)" with "matrix.append(row#)"
    for row1 in kappaRead:
        #reqData1 = [row1[0], row1[1], row1[2]]
        kappaMatrix.append(row1[1])

    for row2 in lambdaRead:
        #reqData2 = [row2[0], row2[1], row2[2]]
        lambdaMatrix.append(row2[1])

    # If each vector needs to be represented as a row, do for each in matrix
    # If each vector needs to be represented as a column, do for each in dataToWrite
    for row3 in kappaMatrix:
        csvWrite.writerow([row3])

    for row4 in lambdaMatrix:
        csvWriteTwo.writerow([row4])

    return [np.array(kappaMatrix), np.array(lambdaMatrix)]

getKappaLambda()
