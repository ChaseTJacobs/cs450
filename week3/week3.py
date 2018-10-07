import sys
import math
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class KNNModel:
	def __init__(self, datatrain, targetstrain):
		self.datatrain = datatrain
		self.targetstrain = targetstrain
		
	def predict(self, k, datatest):
		targettest = []
		for x in datatest:
			votes = {}
			neighbors = self.getNeighbors(x, k)
			for x in range(len(neighbors)):
				vote = neighbors[x][0][1]
				if(votes.get(vote, False)):
					votes[vote] = votes[vote] + 1
				else:
					votes[vote] = 1
			highest = (-1, 0)
			for x,y in votes.items():
				if y > highest[1]:
					highest = (x,y)
			targettest.append(highest[0])
		return targettest
		
	def computeDistance(self, point1, point2, length):
		distance = 0
		for x in range(length):
			distance += pow((point1[x] - point2[x]), 2)
		return math.sqrt(distance)
	
	def getNeighbors(self, point, k):
		distances = []
		length = len(point)
		for x in range(len(self.datatrain)):
			dist = self.computeDistance(point, self.datatrain[x], length)
			distances.append(([self.datatrain[x], self.targetstrain[x]], dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for x in range(k):
			neighbors.append((distances[x][0], distances[x][1]))
		return neighbors

class KNNClassifier:
	def fit(self, xtrain, ytrain):
		model = KNNModel(xtrain, ytrain)
		return model

def findPercentage(predicted, ytest):
	count = 0
	successes = 0
	for x in predicted:
		if x == ytest[count]:
			successes = successes + 1
		count = count + 1
	print ("% of correct guesses: ", float("{0:.2f}".format(successes/len(ytest) * 100)), "% (",successes,"/",len(ytest),")")

def readInCars(k):
	car = pd.read_csv("car.csv")
	car.columns = ["buying", "maint", "doors", "persons", "lugboot", "safety", "values"]
	values = car["values"]
	values = values.values

	car = car.drop(['values'], axis=1)

	car = pd.get_dummies(car)
	car = car.values

	xtrain, xtest, ytrain, ytest = train_test_split(car, values, test_size=.7, random_state=42)
	classifier = KNNClassifier()
	model = classifier.fit(xtrain, ytrain)
	targets_predicted = model.predict(k, xtest)
	print("My percentage: ")
	findPercentage(targets_predicted, ytest)
	
def readInCars2(k):
	car = pd.read_csv("auto-mpg.csv", delim_whitespace=True)
	car.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]
	target = car["origin"]
	target = target.values

	car = car.drop(['origin'], axis=1)

	car = pd.get_dummies(car)
	car = car.values

	xtrain, xtest, ytrain, ytest = train_test_split(car, target, test_size=.7, random_state=42)
	classifier = KNNClassifier()
	model = classifier.fit(xtrain, ytrain)
	targets_predicted = model.predict(k, xtest)
	print("My percentage: ")
	findPercentage(targets_predicted, ytest)
	
def readInAutism(k):
	dataset = pd.read_csv("Autism-Adult-Data.csv")
	dataset.columns = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "age", "gender", "ethnicity", "BWJ", "FM-PDD", "country", "usedApp", "SMT", "conductingTest", "haveAutism"]
	target = dataset["haveAutism"]
	target = target.values
	
	dataset = dataset.drop(["haveAutism"], axis=1)
	dataset = pd.get_dummies(dataset)
	dataset = dataset.values
	
	xtrain, xtest, ytrain, ytest = train_test_split(dataset, target, test_size=.7, random_state=42)
	classifier = KNNClassifier()
	model = classifier.fit(xtrain, ytrain)
	targets_predicted = model.predict(k, xtest)
	print("My percentage: ")
	findPercentage(targets_predicted, ytest)


whichTest = input("Which dataset do you want to test? (1 for car.csv, 2 for auto-mpg.csv and 3 for autism.csv): ")

k = 5
whichTest = int(whichTest)

print("Calculating... Please wait, as this could take a few moments...")
if whichTest == 1:
	readInCars(k)
elif whichTest == 2:
	readInCars2(k)
elif whichTest == 3:
	readInAutism(k)
else:
	print("Input not valid")






