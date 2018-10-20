# Written by Chase Jacobs

import numpy as np
import pandas as pd
import math

from sklearn import datasets
from sklearn.model_selection import train_test_split

class tree:
	def __init__(self, value):
		self.value = value
		self.nodes = []

class decisionTree:
	def __init__(self, data, attributes):
		self.data = data
		self.attributes = attributes
		self.tree = tree(None)
		self.calcRootNode()
		
	def estimateData(self, row, treeNode):
		if len(treeNode.nodes) == 0:
			return treeNode.value
		for node in treeNode.nodes:
			if row[treeNode.value] == node[0]:
				return self.estimateData(row, node[1])
				
	def calculateEntropy(self, yesses, nos, total):
		return (-(yesses/total)*math.log2(yesses/total)-(nos/total)*math.log2(nos/total))
		
	def buildTree(self, parentNode, dataset, attr, optionName):
		# The dataset needs to be "cut down" in rows before being passed in here
#		print("Parent node: ", parentNode.value)
#		print("dataset:", dataset)
#		print("new attributes (branches): ", attr)
#		print("Option Name: ", optionName)
		
		isLeaf = True
		shouldLoan = dataset[['ShouldLoan']].values
		tempAnswer = shouldLoan[0]
		
		for answer in shouldLoan:
			if answer != tempAnswer:
				isLeaf = False
			
		if isLeaf:
			newLeafNode = tree(tempAnswer)
			parentNode.nodes.append((optionName,newLeafNode))
#			print("Create leaf node: ", tempAnswer)
			return 0
#		if len(shouldLoan) == 1:
#			newLeafNode = tree(value)
			
#		if isLeaf:
#		Is a leaf when:
#			all results are the same - set leaf node value = result value
#			only one more attribute - set leaf node value to highest # of results
			
			
		tempDict = {}
		for attribute in attr:
#			print("Starting attribute for loop with: ", attribute)
			tempDict[attribute] = []
			for column in dataset:
#				print("startingg column dataset with: ", column)
				if attribute in column:
					countY = 0
					countN = 0
					for index, x in enumerate(dataset[column]):
#						print("Checking ", column, " is ", x)
						if x == 1:
							if shouldLoan[index]:
								countY = countY + 1
							else:
								countN = countN + 1
#					print("In ", column, " there are ", countY, " yesses and ", countN, " nos")
					total = countN + countY
					if countY == total or countN == total:
						tempEnt = 0
					else:
						tempEnt = self.calculateEntropy(countY, countN, total)
#					print("Entropy for ", column, " is ", tempEnt)
					tempDict[attribute].append(tempEnt * (total/len(dataset)))
#		print("Dictionary: ", tempDict)
		node = ("none", 9999)
		for ent in tempDict:
			average = 0
			for x in tempDict[ent]:
				average = average + x
			if average < node[1]:
				node = (ent, average)
#			print(ent, " average: ", average)
#		print("Lowest: ", node)
		
		newAttribute = list(attributes)
		newAttribute.remove(node[0])
		
		branches = []
		for column in dataset:
			if node[0] in column:
				branches.append(column)
				
		branchNames = list(branches)
		
#		print("Possible branches: ", branches)
		# tree(value, leaf, branches)
		newNode = tree(node[0])
		parentNode.nodes.append((optionName,newNode))
		# loop through branches (options for root node, IE income high/low)
		for branch in branches:
			branchData = dataset
			for index, row in dataset.iterrows():
				if row[branch] == 0:
					branchData = branchData.drop(index, axis=0)
			optionName = branch.replace(node[0], '')
			optionName = optionName.replace('_', '')
#			print("new Branch: ", branchData)
			# Cut down data and attribute list, pass into buildTree(tree, data, remainingAttr)
			self.buildTree(newNode, branchData, newAttribute, optionName)
		
	def calcRootNode(self):
		df1 = self.data[['ShouldLoan']].values
		tempDict = {}
		for attr in self.attributes:
			tempDict[attr] = []
			for column in self.data:
				if attr in column:
					countY = 0
					countN = 0
					for index, x in enumerate(self.data[column]):
						if x == 1:
							if df1[index]:
								countY = countY + 1
							else:
								countN = countN + 1
#					print("In ", attr, " there are ", countY, " yesses and ", countN, " nos")
					total = countN + countY
					if countY == total or countN == total:
						tempEnt = 0
					else:
						tempEnt = self.calculateEntropy(countY, countN, total)
					tempDict[attr].append(tempEnt * (total/len(newData)))
					
		rootNode = ("none", 9999)
		for ent in tempDict:
			average = 0
			for x in tempDict[ent]:
				average = average + x
			if(average < rootNode[1]):
				rootNode = (ent, average)
#			print(ent, " average: ", average)
#		print("Lowest: ", rootNode)
		
		newAttr = self.attributes
		newAttr.remove(rootNode[0])
#		print(newAttr)
		
		branches = []
		
		for column in self.data:
			if rootNode[0] in column:
				branches.append(column)

		baseTree = tree(rootNode[0])
		# loop through branches (options for root node, IE income high/low)
		for branch in branches:
#			print("Branch: ", branch)
			branchData = self.data
			for index, row in self.data.iterrows():
				if row[branch] == 0:
					branchData = branchData.drop(index, axis=0)
			# Cut down data and attribute list, pass into buildTree(tree, data, remainingAttr)
			optionName = branch.replace(rootNode[0], '')
			optionName = optionName.replace('_', '')
			self.buildTree(baseTree, branchData, newAttr, optionName)
		
		self.tree = baseTree
	
#	one node at a Time
#	filter data based on the data that hasn't been asked
#	Recursive Function
#	pseudocode available (for each attribute do the following)
	
#	calculate the entropy of each choice
#	Create node, move on to next ones.
	
iris = datasets.load_iris()
xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target, test_size=0.7, random_state=42)

data = [["Good"		,"High"	,"Good"	, True], 
		["Good"		,"High"	,"Poor"	, True], 
		["Good"		,"Low"	,"Good"	, True],
		["Good"		,"Low"	,"Poor"	, False],
		["Average"	,"High"	,"Good"	, True],
		["Average"	,"Low"	,"Poor"	, False],
		["Average"	,"High"	,"Poor"	, True],
		["Average"	,"Low"	,"Good"	, False],
		["Low"		,"High"	,"Good"	, True],
		["Low"		,"High"	,"Poor"	, False],
		["Low"		,"Low"	,"Good"	, False],
		["Low"		,"Low"	,"Poor"	, False]]
attributes = ["CreditScore", "Income", "Collateral"]

data = pd.read_csv("week5.csv")
dataTest = pd.read_csv("target.csv")
data.columns = ["CreditScore", "Income", "Collateral", "ShouldLoan"]
dataTest.columns = ["CreditScore", "Income", "Collateral", "Answer"]
newData = pd.get_dummies(data)

def displayNodes(tree):
	print("Tree Value: ", tree.value)
	for node in tree.nodes:
		print(tree.value, " branch: ", node[0])
		displayNodes(node[1])

decisionTreeClassifier = decisionTree(newData, attributes)
displayNodes(decisionTreeClassifier.tree)

answers = dataTest[['Answer']].values
correctCounter = 0
for index, row in dataTest.iterrows():
	estimate = decisionTreeClassifier.estimateData(row, decisionTreeClassifier.tree)
	print("Estimate for row ", index, " is: ", estimate)
	print("The correct answer is: ", answers[index])
	if estimate == answers[index]:
		correctCounter = correctCounter + 1

print("Succesfully estimated ", (correctCounter / len(answers)) * 100, "%")
