import csv

def getVectors():
	kappaData = open("KappaFLC.csv", "r")
	reader = csv.reader(kappaData)
	patientRow = []
	patientVector = []
	kappaData = open("KappaFLC.csv", "r")
	reader = csv.reader(kappaData)
	modifiedDate = 0
	characterCount = 0
	rowCount = -1
	for row in reader:
		if(rowCount > -1):	
			patientVector.append([row[0].strip("MM-"), row[1], '', ''])
			date = row[2]
			if(len(date) == 13):
				date = "0" + date[:2] + "0" + date[2:]
			if(len(date) == 14):
				if "/" in date[:2]:
					date = "0" + date
				else:
					date = date[:3] + "0" + date[3:]
			date = date.replace("/", "").replace(" 0:00", "")
			patientVector[rowCount][3] = date
		rowCount += 1
	lambdaData = open("LambdaFLC.csv", "r")
	reader = csv.reader(lambdaData)
	rowCount = 0
	for row in reader:
		patientVector[rowCount][2] = row[1]
		rowCount += 1
	rowCount = 0
getVectors()