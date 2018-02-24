import csv

def getVectors():
	kappaData = open("KappaFLC.csv", "r")
	reader = csv.reader(kappaData)
	patientRow = []
	patientVector = []
	kappaData = open("KappaFLC.csv", "r")
	reader = csv.reader(kappaData)
	for row in reader:
		patientVector.append([row[0], row[1], '', row[2]])
	lambdaData = open("LambdaFLC.csv", "r")
	reader = csv.reader(lambdaData)
	rowCount = 0
	for row in reader:
		patientVector[rowCount][2] = row[1]
	return patientVector