# Neural Network Assignment - Week 8
# Written by Chase Jacobs

import math
import numpy as np
import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Each node contains certain variables
# value - This is "a" in terms of our reading and equations given.
# weights - This is a list of weights that correspond to the nodes on the previous layer. Randomized.
# error - This is the error value. Default is 0 but we go back and calculate it during back propogation.
# isBiasNode - This is an easy way to keep track whether a node is a bias node or not, as
#			   bias nodes don't have weights that connect to the previous layer.
class NeuralNode:
	def __init__(self, value, prevLength, isBiasNode):
		self.value = value
		self.weights = []
		self.error = 0
		self.isBiasNode = isBiasNode
		if isBiasNode == False:
			for x in range(prevLength):	
				self.weights.append(random.uniform(-1,1))
		

class NeuralNetwork:
	def __init__(self, inputValues, expectedOutput, hiddenLayersArray, nodesInOutputLayer):
		print("Creating neural network with ", len(inputValues[0]), " input nodes, ", len(hiddenLayersArray), " hidden layers, and ", nodesInOutputLayer, " nodes in the output layer.")
		self.inputValues = inputValues
		self.expectedOutput = expectedOutput
		self.hiddenLayersArray = hiddenLayersArray
		self.networkLayers = []
		
		# add bias node to input layer with value -1
		inputLayer = [(NeuralNode(-1, False, True))]
		
		lastArray = len(inputValues[0])
		for x in range(lastArray):
			inputLayer.append(NeuralNode(False, False, False))
		self.networkLayers.append(inputLayer)
		
		# Create hidden layers
		hiddenLayers = []
		lastArray = len(inputLayer)
		for x in hiddenLayersArray:
#			Start with bias node with value -1
			layer = [(NeuralNode(-1, lastArray, False))]
			for nodes in range(x):
				layer.append(NeuralNode(False, lastArray, False))
			lastArray = len(layer)
			self.networkLayers.append(layer)
		
		# Create output layer
		outputLayer = []
		for j in range(nodesInOutputLayer):
			outputLayer.append((NeuralNode(False, lastArray,False)))
		self.networkLayers.append(outputLayer)		
		
	def calculateResult(self, nInput):
		resultArray = []
		for index, layer in enumerate(self.networkLayers):
			if index == 0:
				for index2, node in enumerate(layer):
					if index2 != 0:
						node.value = nInput[index2-1]
			else:
				for nodeIndex, node in enumerate(layer):
					if node.isBiasNode == False:	
						sumTotal = 0
						for weightindex, weight in enumerate(node.weights):
							sumTotal = sumTotal + (weight * self.networkLayers[index - 1][weightindex].value)
						#add together all weights and values from previous node and run it through sigmoid
						node.value = self.sigmoid(sumTotal)
						if index == len(self.networkLayers) - 1:
							resultArray.append(node.value)
		return resultArray
	
#	Sigmoid function.
	def sigmoid(self, h):
		return (1 / (1 + math.exp(-h)))
		
	def backProp(self, expectedResult, result, learningRate):
		# loop through each layer and calculate error of each node
		errors = []
		for i, layer in reversed(list(enumerate(self.networkLayers))):
			if i == len(self.networkLayers) - 1:
				# This is the output layer
				for nodeIndex, node in enumerate(layer):
					node.error = node.value * (1 - node.value) * (node.value - expectedResult[nodeIndex])
					errors.append(node.error)
			elif i == 0:
				# This is the first layer. Skip the back propogation for this layer.
				yes = 1
			else:
				# This is every other layer
				for indexNode, node in enumerate(layer):
					sumOfErrors = 0
					for prevIndex, previousNode in enumerate(self.networkLayers[i + 1]):
						if prevIndex != 0 or (prevIndex == 0 and i == len(self.networkLayers) - 2):
							sumOfErrors = sumOfErrors + (previousNode.error * previousNode.weights[indexNode])
					node.error = node.value * (1 - node.value) * (sumOfErrors)
				
		# loop through again and change weights based on errors
		for layerIndex, layer in enumerate(self.networkLayers):
			if layerIndex != 0:
				for nodeIndex, node in enumerate(layer):
#					print("Before node weights: ", node.weights)
					newWeights = []
					for weightIndex, weight in enumerate(node.weights):
#						print(learningRate, node.error, node.value)
						newWeights.append(weight - (learningRate * node.error * self.networkLayers[layerIndex - 1][weightIndex].value))
					node.weights = list(newWeights)
		return errors
			
	
print("Starting neural network tests. This takes a few minutes due to multiple tests on the iris dataset with a large number of training iterations.")
x1   = [[1.2, .6],
		[1.9, 2.1],
		[-1, 0.05],
		[0.2,-0.2]]
y = [[1],[0],[1],[1]]

# The hidden layers array is how we define how many hidden layers
# there are and how many nodes are in each layer. For each integer
# in the array there is a hidden layer, and that integer defines
# how many nodes are in that layer.
hiddenLayers = [2,4,8,8]
epoch = 5000
print("Training with simple data set with ", epoch, " iterations.")
network = NeuralNetwork(x1, y, hiddenLayers, 1)

# Train simple data set
for epach in range(epoch):
	for x in range(len(x1)):
		result = [network.calculateResult(x1[x])]
		network.backProp(y[x], result, .1)

# Test simple data set. Also add another point to see the output.
for x in range(len(x1)):
	result = [network.calculateResult(x1[x])]
	print("For ", x1[x], " the result is : ", result, " and the expected output is: ", y[x])
result = network.calculateResult([-1,0])
print("At data point (-1,0) the predicted result is: ", result)
	
	

# This function serves two purposes. First, it can find the result
# of neural network based on the given arrays by subtracting the result
# by 1. So if the 3 output nodes give [.01, .97, .02] then the correct
# classifier is the 2nd one. This function will return that classifier.
# It also serves a purpose to undo the preperation of the result data.
# For example, I turned each expected output of "1" to [0,1,0] so it
# could be used to calculate the error of the 3 output nodes. This
# function will turn [0,1,0] back to "1" as the same calculation
# works.
def findResults(possibleResults, results):
	highestDifference = 1
	returnIndex = len(results)
	for index, result in enumerate(results):
		tempDiff = 1 - result
		if tempDiff < highestDifference:
			highestDifference = tempDiff
			returnIndex = index
	return possibleResults[returnIndex]
	
def testIris(epoch, hiddenLayers, graphNumber):
	iris = datasets.load_iris()
	xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target, test_size=0.7, random_state=42)

	# Prepare iris datasets
	tempYtrain = []
	tempYtest = []
	for y in ytrain:
		temp = []
		if y == 0:
			temp = [1,0,0]
		elif y == 1:
			temp = [0,1,0]
		elif y == 2:
			temp = [0,0,1]
		tempYtrain.append(temp)
	ytrain = tempYtrain
	for y in ytest:
		temp = []
		if y == 0:
			temp = [1,0,0]
		elif y == 1:
			temp = [0,1,0]
		elif y == 2:
			temp = [0,0,1]
		tempYtest.append(temp)
	ytest = tempYtest
	#xtrain matches with ytrain
	#xtest matches with ytest

	possibleResults = [0,1,2]
	print("Training with iris data set with ", epoch, " iterations.")
	network2 = NeuralNetwork(xtrain, ytrain, hiddenLayers, 3)
	errorOverTime = []
	# Train the network with the xtrain and ytrain part of the data set (70% of original dataset)
	for epoch2 in range(epoch):
		for y in range(len(xtrain)):
			result2 = network2.calculateResult(xtrain[y])
			error = network2.backProp(ytrain[y], result2, .1)
			error = (sum(error) / len(error))
			errorOverTime.append(error)
			
			
	# Here is where I graph the error. There is an error recorded for every
	# time we back propogate. Meaning the x axis will always be
	# (# of epocs * number of data). For the 1st and 3rd tests the error
	# smoothes out over time (hence the high success rate of the network)
	# but in the 2nd one it doesn't (hence the low success rate). This is probably
	# due to low amount of epocs.
	fig = plt.figure(graphNumber)
	t = np.linspace(0, len(errorOverTime), len(errorOverTime), endpoint=True)
	plt.plot(t, errorOverTime, 'b-', label='error')
	plt.legend(loc = 'upper right')
	plt.xlabel('# of times backpropagated')
	plt.ylabel('y')
	fig.savefig("irisErrorPlot" + str(graphNumber))
			
	correctCounter = 0
	for y in range(len(xtest)):
		result2 = network2.calculateResult(xtest[y])
		actualResult = findResults(possibleResults, result2)
		expectedResult = findResults(possibleResults, ytest[y])
		if expectedResult == actualResult:
			correctCounter = correctCounter + 1
#		print("For ", xtest[y], " the result is: ", actualResult, " and the expected output is: ", expectedResult)

	print("For the iris dataset with hidden arrays: ",hiddenLayers , "and", epoch ,"iterations we correctly predict the data of the testing set ", ((correctCounter / len(xtest)) * 100), "% of the time.")
	
epoch = 2000
hiddenLayers2 = [2,4,3]
testIris(epoch, hiddenLayers2, 2)

epoch = 1000
hiddenLayers2 = [1,2,3,4,5]
testIris(epoch, hiddenLayers2, 3)

epoch = 5000
hiddenLayers2 = [8,8,8,2]
testIris(epoch, hiddenLayers2, 4)
# Test the neural network with the xtest and ytest part of the data set (30% of the original dataset)

	




