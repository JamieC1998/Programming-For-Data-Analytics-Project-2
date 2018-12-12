import pandas as pd
import time
from collections import Counter
import matplotlib.pyplot as plt;
import re
from math import log
plt.rcdefaults
from nltk.corpus import stopwords

# -----------File Directories------------
trainPosDirectory = "train/trainPos.txt"
trainNegDirectory = "train/trainNeg.txt"

testPosDirectory = "test/testPos.txt"
testNegDirectory = "test/testNeg.txt"
# ----------------------------------------

# ------------Global Vars-----------------

#When this flag is set to True
#It will enable PreProcessing
PRE_PROCESSING = True

#When this flag
DEBUG = False
# ----------------------------------------

def main():
    print("UNPROCESSED")
    unprocessedAccuracy = process_data(False)
    print("PROCESSED")
    processedAccuracy = process_data(True)
    plotGraph(unprocessedAccuracy, processedAccuracy)
    

def process_data(boolVal):
    global PRE_PROCESSING
    PRE_PROCESSING = boolVal
    return calculateTotalAccuracy()

def calculateTotalAccuracy():
    startTime = time.time()

    # -----------------------Training-------------------------
    trainStartTime = time.time()

    dataPos = createDictionary(trainPosDirectory)
    dataNeg = createDictionary(trainNegDirectory)

    occuranceOverall = getOverallOccurance(dataPos, dataNeg)

    positiveProbability = calculateProbability(occuranceOverall, dataPos)
    negativeProbability = calculateProbability(occuranceOverall, dataNeg)

    trainFinishTime = time.time()
    trainTime = trainFinishTime - trainStartTime
    # -----------------------Training-------------------------
    classStartTime = time.time()

    posAccuracy = determineAccuracy(positiveProbability, negativeProbability, testPosDirectory, True)
    negAccuracy = determineAccuracy(positiveProbability, negativeProbability, testNegDirectory, False)

    print(f"\tPositive File:\n\t\tPositive Line Count: {posAccuracy[0]}\n\t\tNegative Line Counter: {posAccuracy[1]}")
    print(f"\tNegative File:\n\t\tPositive Line Count: {negAccuracy[0]}\n\t\tNegative Line Count: {negAccuracy[1]}")
    
    posAccuracyNum = posAccuracy[0] / (posAccuracy[0] + posAccuracy[1])
    negAccuracyNum = negAccuracy[1] / (negAccuracy[1] + negAccuracy[0])

    classFinishTime = time.time()
    classTime = classFinishTime - classStartTime

    endTime = time.time()
    print(f"\n\tPositive Accuracy\t: {posAccuracyNum}")
    print(f"\tNegative Accuracy\t: {negAccuracyNum}")
    print(f"\n\tOverrall Accuracy\t: {((posAccuracy[0] + negAccuracy[1]) / 2)/10}")
    print(f"\tTime\t\t\t: {round(endTime - startTime)}s")

    return ((posAccuracy[0] + negAccuracy[1]) / 2) / 10

def plotGraph(unprocessedAccuracy, processedAccuracy):
    y_axis = [0, 100]
    dataNames = ('Without Pre-Processing', 'Pre-Processed')
    itemsToPlot = [unprocessedAccuracy, processedAccuracy]

    plt.bar(y_axis, itemsToPlot, align='center', alpha=1)
    plt.xticks(y_axis, dataNames)

    plt.ylabel("Accuracy")
    plt.xlabel("Data")

    plt.title("Pre-Processing vs. No Pre-Processing")
    plt.show()

def determineAccuracy(posProb, negProb, fileName, whichFile):
    fp = open(fileName)
    data = fp.read().splitlines()

    posTweetCounter = 0
    negTweetCounter = 0

    if DEBUG == True:
        print("Classifying Tweets")

    for tweets in data:
        posWordProb = 1
        negWordProb = 1
        
        tweet = tweets
        
        if PRE_PROCESSING == True:
            tweet = cleanString(tweets.lower())

        for word in tweet.split():
            if word in posProb.keys():
                posWordProb *= posProb[word]

            if word in negProb.keys():
                negWordProb *= negProb[word]

        if posWordProb > negWordProb:
            posTweetCounter += 1
            if DEBUG == True:
                print(f"{tweet}:\tPositive")
        else:
            negTweetCounter += 1
            if DEBUG == True:
                print(f"{tweet}:\tNegative")

    return [posTweetCounter, negTweetCounter]

def calculateProbability(occuranceOverall, dataSet):
    probability = dict.fromkeys(occuranceOverall.keys(), 0)

    for each in probability.keys():
        if each in dataSet.keys():
            probability[each] = dataSet[each] / occuranceOverall[each]
            if DEBUG == True:
                print(f"{each}:\t{probability[each]}")
            

    return probability

def cleanString(the_string):
    the_clean_string = re.sub(r"[\;\?\<\>\,\"\|\:\.\`\~\{\}\\\/@$\%\[\]\(\)\^\-\+\&#\*\!\_\=]*", "", the_string)

    return the_clean_string

def getOverallOccurance(dataPos, dataNeg):
    completeSet = set()

    completeSet.update(dataPos.keys(), dataNeg.keys())

    occuranceOverall = dict.fromkeys(completeSet, 0)

    occuranceOverall = calculateOccurance(occuranceOverall, dataPos, dataNeg)

    return occuranceOverall


def calculateOccurance(occuranceOverall, dataPos, dataNeg):
    if DEBUG == True:
        print("Overall Occurance")
    
    for each in occuranceOverall.keys():

        if each in dataPos.keys():
            occuranceOverall[each] += dataPos[each]

        if each in dataNeg.keys():
            occuranceOverall[each] += dataNeg[each]

        if DEBUG == True:
            print(f"\n{each}:\t{occuranceOverall[each]}")

    return occuranceOverall


def createDictionary(fileName):
    # Open the file
    fp = open(fileName)

    # Reading it in
    fileRead = fp.read().lower()

    fileList = fileRead.split()

    if PRE_PROCESSING == True:
        for i in range(0, len(fileList)):
            fileList[i] = cleanString(fileList[i])

    if PRE_PROCESSING == True:
        fileList = [word for word in fileList if word not in stopwords.words('english')]

    myDict = dict.fromkeys(set(fileList), 0)

    for word in fileList:
        myDict[word] += 1

    if DEBUG == True:
        print("Creating Dictionary")
        count = 0
        for each in myDict.keys():
            if count < 25:
                print(f"{each}:\t{myDict[each]}")
                count += 1

    return myDict


if __name__ == "__main__":
    main()