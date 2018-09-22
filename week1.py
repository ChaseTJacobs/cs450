import sys
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

class HardCodedModel:
	def __init__(self, datatrain, targetstrain):
		self.datatrain = datatrain
		self.targetstrain = targetstrain
		
	def predict(self, datatest):
		targettest = []
		for x in datatest:
			targettest.append(0)
		return targettest

class HardCodedClassifier:
	def fit(self, xtrain, ytrain):
		model = HardCodedModel(xtrain, ytrain)
		return model
		
def findPercentage(predicted, ytest):
	count = 0
	successes = 0
	for x in predicted:
		if x == ytest[count]:
			successes = successes + 1
		count = count + 1
	print ("% of correct guesses: ", float("{0:.2f}".format(successes/len(ytest) * 100)), "% (",successes,"/",len(ytest),")")
	
trainPer = input("Enter % of data to be used to train: ")
whichModal = input("Enter 1 to use Gaussian prediction, and 2 to use my hardcoded prediction: ")

trainPer = int(trainPer)
whichModal = int(whichModal)
if trainPer <= 0 or trainPer >= 100:
	print("Invalid training percentage. Reverting to default of 70%")
	trainPer = 3
trainPer = 1 - (trainPer / 100)

if whichModal < 1 or whichModal > 2:
	print("Invalid selection for prediction method. Reverting to default of 1")
	whichModal = 1
	
iris = datasets.load_iris()

# Show the data (the attributes of each instance)
#print("the data: ", iris.data)

# Show the target values (in numeric format) of each instance
#print("The target: ", iris.target)

# Show the actual target names that correspond to each number
#print("The name: ", iris.target_names)

xtrain, xtest, ytrain, ytest = train_test_split(iris.data, iris.target, test_size=trainPer, random_state=42)
													
#print ("xtrain: ", xtrain)
#print ("xtest: ", xtest)
#print ("ytrain: ", ytrain)
#print ("ytest: ", ytest)

if whichModal == 1:
	classifier = GaussianNB()
	model = classifier.fit(xtrain, ytrain)

	targets_predicted = model.predict(xtest)

	print("Gaussian Estimate:")
	findPercentage(targets_predicted, ytest)
elif whichModal == 2:
	classifier2 = HardCodedClassifier()
	model2 = classifier2.fit(xtrain, ytrain)
	targets_predicted2 = model2.predict(xtest)

	print("My Hardcoded Estimate:")
	findPercentage(targets_predicted2, ytest)