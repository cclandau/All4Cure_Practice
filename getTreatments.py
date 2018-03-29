import csv
import numpy as np
from datetime import datetime
np.set_printoptions(threshold=np.nan)

def getTreatments():

	patList = open("PatientList.csv", "r")
	reader = csv.reader(patList)
	patList = np.array(list(reader))
	patList = np.delete(patList, np.s_[0], axis=0)
	numPats = patList.shape[0]

	treatList = open("LabsTreatmentsList.csv", "r")
	reader = csv.reader(treatList)
	treatList = np.array(list(reader))
	treatList = np.delete(treatList, np.s_[0], axis=0)

	treatDict = {}
	for i in range(0, treatList.shape[0] - 1):
		if treatList[i][1] != "Lab":
			treatList[i][5] = treatList[i][5].replace(" 00:00:00", "")
			date = datetime.strptime(treatList[i][5], '%Y-%m-%d').date()
			if treatList[i][0] in treatDict:
				treatDict[treatList[i][0]].append([treatList[i][2], date])
			else:
				temp = []
				treatDict[treatList[i][0]] = temp
				treatDict[treatList[i][0]].append([treatList[i][2], date])

	for i in treatDict:
		temp = treatDict[i]
		treatDict[i] = sorted(temp, key=lambda temp_entry: temp_entry[1])
	print(treatDict["MM-1"])
	return treatDict

getTreatments()
