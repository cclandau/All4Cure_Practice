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

def pullRows(inFilename, outFilename, rowsFilename, matrix):
    global fullMatrix
    global rows
    benchMatrix = []
    with open(inFilename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        fullMatrix = list(csvfile)
    with open(rowsFilename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(csvfile)
    out = []
    for i in range(len(rows)):
        index = int(rows[i].replace('\n', '')) - 1
        out.append(fullMatrix[index])
        benchMatrix.append(matrix[index, :])
    with open(outFilename, 'w') as csvfile:
        for i in range(len(out)):
            csvfile.write(out[i])
    return(np.array(benchMatrix))
