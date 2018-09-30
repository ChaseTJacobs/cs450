# Week 2 Ponder
# Written by Chase Jacobs

import sys
import math
import operator
from random import randint
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
	
def knn(k, iris):
	print("K = ", k)
	scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
	
	scaler.fit(iris.data)
	
	xtrain, xtest, ytrain, ytest = train_test_split(scaler.transform(iris.data), iris.target, test_size=70, random_state=randint(1,100))
	
	classifier = KNNClassifier()
	model = classifier.fit(xtrain, ytrain)
	targets_predicted = model.predict(k, xtest)
	print("My percentage: ")
	findPercentage(targets_predicted, ytest)
	
	classifier2 = KNeighborsClassifier(n_neighbors=k)
	model2 = classifier2.fit(xtrain, ytrain)
	predictions2 = model2.predict(xtest)
	print("sklearn percentage: ")
	findPercentage(predictions2, ytest)
	return 0
	
iris = datasets.load_iris()

knn(3, iris)
knn(5, iris)
knn(7, iris)
knn(30, iris)