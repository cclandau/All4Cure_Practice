import csv
import numpy as np
from datetime import datetime
np.set_printoptions(threshold=np.nan)

def getVectors():

	#Read from the list of patients and create a dictionary saying which patient
	#is Kappa and which is Lammbda
	patList = open("PatientList.csv", "r")
	reader = csv.reader(patList)
	patList = np.array(list(reader))
	patDict = {}
	for i in range(1, patList.shape[0] - 1):
		patDict[patList[i][0]] = patList[i][3]

	#Make matrices of FLC values and delete unneccessary information
	kappaData = open("KappaFLC.csv", "r")
	reader = csv.reader(kappaData)
	kapFLC = np.array(list(reader))

	lambdaData = open("LambdaFLC.csv", "r")
	reader = csv.reader(lambdaData)
	lamFLC = np.array(list(reader))
	
	kapFLC = np.delete(kapFLC, np.s_[0], axis = 0)
	lamFLC = np.delete(lamFLC, np.s_[0], axis = 0)
	kapFLC = np.delete(kapFLC, np.s_[3,4], axis = 1)
	lamFLC = np.delete(lamFLC, np.s_[3,4], axis = 1)

	#Loop through Kappa patient records, deleting any records of Lambda patients
	#At the same time, read the date into a list of datetime objects for better use
	temp1 = 0
	temp2 = 0
	kapDates = []
	lamDates = []
	while temp1 != kapFLC.shape[0]:
		if patDict.get(kapFLC[temp1][0]) != "Kappa":
			kapFLC = np.delete(kapFLC, np.s_[temp1], axis = 0)
		else:
			kapFLC[temp1][2] = kapFLC[temp1][2].replace(" 0:00", "")
			if len(kapFLC[temp1][2]) == 8:
				kapFLC[temp1][2] = "0" + kapFLC[temp1][2]
				kapFLC[temp1][2] = kapFLC[temp1][2][0:3] + "0" + kapFLC[temp1][2][3:]
			elif len(kapFLC[temp1][2]) == 9:
				if kapFLC[temp1][2][0:2] == "11" or kapFLC[temp1][2][0:2] == "12" or kapFLC[temp1][2][0:2] == "10":
					kapFLC[temp1][2] = kapFLC[temp1][2][0:3] + "0" + kapFLC[temp1][2][3:]
				else:
					kapFLC[temp1][2] = "0" + kapFLC[temp1][2]
			kapDates.append(datetime.strptime(kapFLC[temp1][2], '%m/%d/%Y').date())
			temp1 = temp1 + 1

	#Do the same thing for the Lambda patient records
	while temp2 != lamFLC.shape[0]:
		if patDict.get(lamFLC[temp2][0]) != "Lambda":
			lamFLC = np.delete(lamFLC, np.s_[temp2], axis = 0)
		else:
			lamFLC[temp2][2] = lamFLC[temp2][2].replace(" 00:00:00", "")
			lamDates.append(datetime.strptime(lamFLC[temp2][2], '%Y-%m-%d').date())
			temp2 = temp2 + 1

	#Delete the old date column and return
	kapDates = np.array(kapDates)
	lamDates = np.array(lamDates)
	kapFLC = np.delete(kapFLC, np.s_[2], axis = 1)
	lamFLC = np.delete(lamFLC, np.s_[2], axis = 1)
	return [kapFLC, kapDates, lamFLC, lamDates]

getVectors()
