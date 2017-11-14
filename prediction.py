import sys
import glob
import nltk
from nltk import stem
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.corpus import sentiwordnet as swn
from collections import Counter
from random import randrange
from sklearn.neighbors import NearestNeighbors
from sklearn import svm

import numpy as np
from functools import reduce
from datetime import timedelta
import time
import os
import csv

from datetime import timedelta
import datetime


def getMagazinePattern(content):

    # 219.txt MEDIA-With iPad sales falling, Apple pushes into businesses - WSJ38 --> Problem

    magazineList = ['(Reuters) -', '(Reuters Breakingviews) -', 'BUZZ', '(IFR) -', 'S&P', 'Reuters US Domestic News Summary', \

                        'PHILIPPINES PRESS', 'MEDIA-', 'What to Watch in the Day Ahead', '(Variety.com)', 'All Rights Reserved.']

    magazineDictionary = {'(Reuters) -' : '(Reuters) -', '(Reuters Breakingviews) -' : '(Reuters Breakingviews) -', \

                          'BUZZ' : 'All Rights Reserved.', '(IFR) -' : '(IFR) -', 'S&P' : 'All Rights Reserved.' , \

                          'Reuters US Domestic News Summary' : 'Following is a summary of current US domestic news briefs.', \

                          'PHILIPPINES PRESS' : 'All Rights Reserved.', 'MEDIA-' : 'MEDIA-', '(Variety.com)' : '(Variety.com)' , \

                          'What to Watch in the Day Ahead' : ')', 'All Rights Reserved.' : 'All Rights Reserved.'}



    for magazine in magazineList:

        if magazine in content:

            return magazineDictionary[magazine]


def findBetween(full, first, last):
    start = full.index(first)+len(first)
    end = full.index(last,start)
    #print (str(start)+"...."+str(end))
    return full[start:end]


# get union of all the kNN of input articles
def createkNNData(k, seedArticles, sentimentData):
    neighbors = NearestNeighbors(n_neighbors = k, algorithm='ball_tree').fit(sentimentData)
    _, indices = neighbors.kneighbors(seedArticles)
    indices = reduce(np.union1d, indices).tolist()    
    return indices


# To extract the article content and save as [date, content]

def extractContent(fileIn):
    article = []
    pat2 = 'Released:'
    pat3 = '.000Z'
    content = ""
        
    with open(fileIn) as infile:
        for line in infile:
            content += line
    
    text = ''
    date = ''

    pat1 = getMagazinePattern(content)
    
    if 'Factiva' in content:
        text = ''
        date = ''
    else:
        text = findBetween(content,pat1,pat2)
        date = findBetween(content,pat2,pat3)
        date = date[:date.rfind('T')]
    
    article = [date,text]        
    return article



def fileProcess(filename, article):  
    tokenizer = RegexpTokenizer(r'\w+')
    snowb = stem.SnowballStemmer("english")
    
    date = article[0]
    text = article[1]
    lines = text.split(".")

    #print article
    content = ""+str(date)+"\n"

    for line in lines:
        # Tokenize 
        words = tokenizer.tokenize(line)
        # Remove stop words
        words = [w for w in words if w not in stopwords.words('english')]
        # Convert to lower case
        words = [w.lower() for w in words]

        for w in words:
            content += w+" "

        content += "\n"


    filename = filename.replace("indArticle", "processArt") 
    with open(filename, 'w') as outfile:
        outfile.write(content)







def computeSentiScoreForSentence(words):

    num = len(words)
    sentiValue = np.zeros(num)

    for i in range(num):
        try:
            
            score = list(swn.senti_synsets(words[i]))[0]

            pos = score.pos_score()
            neg = score.neg_score()

            if (pos+neg) == 0:
                sentiValue[i] = 0.0                
            else:
                sentiValue[i] = float(pos-neg) / float(pos+neg)
        
        except:
            continue

    return sentiValue


def getSVM(train, result, query):
    trainData = np.array(train)
    trainResult = np.array(result)
    clf = svm.SVC()
    clf.fit(trainData, trainResult)

    return clf.predict(query)[0]


def getPriceChange(marketData, row, pastIndices, today):
    priceChange = []
    for idx in pastIndices:
        t0 = marketData[row][idx].astype(np.float)
        t_1 = marketData[row][idx - 1].astype(np.float)
        change = (t0 - t_1) / t0
        priceChange.append(change)
    
    return priceChange

def getLabel(marketData, row, today, futureIndices):
    t0 = float(marketData[row][today[0]])
    t1 = marketData[row][futureIndices]
    t1 = [float(x) for x in t1]
    t1 = np.average(t1)
    
    
    if t0 < t1:
        label = 1.0
    else:
        label = -1.0
    return label

def getColumnIndices(targetDate, allDateList, pastInterval, futureInterval):
    pastIndices = []
    futureIndices = []
   
    today = -1
    for marketDate in allDateList:
        today = today + 1
        if (datetime.datetime.strptime(marketDate, "%d/%m/%Y").date() >= datetime.datetime.strptime(targetDate, "%Y-%m-%d").date()):
            break
    
    today = today +1 

    for i in range(pastInterval):
        pastIndices.append(today - i - 1)
        
    for i in range(futureInterval):
        futureIndices.append(today + i + 1)
    
    pastIndices.reverse()
    return  pastIndices, [today], futureIndices




def main():

    company = sys.argv[1]
    industry = sys.argv[2]

    data = []
    with open('./mktcap3_withHeader.csv', 'rb') as f:
        reader = csv.reader(f)
        data = list(reader)

    allDateList = data[0][1:]
    marketData = np.array(data[1:])

    companyDict = {}
    for i in range(len(marketData)):
        companyDict[marketData[i][0]] = i


    priceFeatureNo = 3
    futurePriceNo = 3



    #remove old files
    oldProcessArt = glob.glob('./queryArt/processArt/*.txt')
    for f in oldProcessArt:
        os.remove(f)
    
    oldSentiArt = glob.glob('./queryArt/artSentiment/*.txt')
    for f in oldSentiArt:
        os.remove(f)

    print("old files removed")



    #read all topics from topics.txt

    print("reading all topics")
    try:

        topicList = []

        industryTopicFile = "./trainingArt/"+industry+"/topics.txt"
        with open(industryTopicFile) as topicsFile:

            for line in topicsFile:

                topicList.extend(line.split())


    except:

        print("topics read error")

    print("finished reading all topics")


    #prepare sentiment and price data from corresponding industry

    print("reading all sentiment data")
    sentimentData = []
    priceData = []

    numberOfTopics = len(topicList)
        
    industryArtSenti = './trainingArt/'+industry+'/artSentiment/*.txt'
    sentiList = glob.glob(industryArtSenti)
    for artSenti in sentiList:

        sentiValues = []
        lineNo = 1
        with open(artSenti) as sentiIn:
            for line in sentiIn:
                if lineNo == 1:
                    lineNo += 1
                    priceData.append(float(line))
                else:
                    values = line.split()
                    for value in values:
                        sentiValues.append(float(value))

        sentimentData.append(sentiValues)    

    print("finished reading all sentiment data")
    
    size = len(sentimentData)

    if size <= 200:
        k = int(0.6*size)
    else :
        k = int(0.4*size)

    #START
    predictionResults = []
    expectedResults = []


    queryArtList = glob.glob('./queryArt/indArticle/*.txt')

    n=1
    for queryArticle in queryArtList:

        #process into proper format
        try:

            print("start processing file "+str(queryArticle))
            articleContent = extractContent(queryArticle)
            print "content extracted"

            fileProcess(queryArticle, articleContent)

            print ("finish processing file "+queryArticle)
           
        except:
            print("File processing error")


   
        #start computing sentiment score for query article
        filename = queryArticle.replace("indArticle", "processArt")
     
        sentiScores = {}
        priceFeature = []

        print("start sentiment computation")

        with open(filename) as infile:

            for line in infile:
                if "-" in line:
                    date=line.strip()

                    row = companyDict[company]

                    pastIndices, today, futureIndices = getColumnIndices(date, allDateList, priceFeatureNo, futurePriceNo)

                    label = getLabel(marketData, row, today, futureIndices)

                    expectedResults.append(label)

                    priceFeature = getPriceChange(marketData, row, pastIndices, today)

                    continue


                words = line.split()
                lineSentiScores = computeSentiScoreForSentence(words)
                

                length = len(words)


                for i in range(length):
                
                    if words[i] in topicList:

                        senti = 0.0
                        for j in range(length):
                            if j!= i:
                                senti += lineSentiScores[j] / abs(j-i)

                        if words[i] in sentiScores:
                            score, count = sentiScores[words[i]]
                            newScore = (score*count + senti) / (count+1)
                            newCount = count + 1
                            sentiScores[words[i]] = [newScore, newCount]
                        else:
                            sentiScores[words[i]] = [senti, 1]


        #finished finding all topic words and sentiment scores
        
        #store sentiment scores in articleSentiment
        print "final sentiScores: ", sentiScores
        articleSentiment = np.zeros(len(topicList))
        topicsFound = sentiScores.keys()
        for topic in topicsFound:
            topicIdx = topicList.index(topic)
            articleSentiment[topicIdx] = sentiScores[topic][0]
        articleSentiment = np.append(articleSentiment, priceFeature)


        #save the sentiment values in txt
        filename = filename.replace("processArt", "artSentiment")    
        with open(filename, 'w') as outfile:
            outputStr = ""
            for v in articleSentiment:
                outputStr += str(v)+" "

            outfile.write(outputStr)

        print ("finished sentiment score of file "+queryArticle)
       
        
        #KNN

        query = []
        query.append(articleSentiment)
        query = np.array(query)

        kNNIndices = createkNNData(k, query, sentimentData)

        print("kNN result:")
        print(kNNIndices)

        #SVM
        trainSentiData = []
        trainPriceResult = []
        for index in kNNIndices:
            trainSentiData.append(sentimentData[index])
            trainPriceResult.append(priceData[index])


        result = getSVM(trainSentiData, trainPriceResult, query)


        #save prediction result
        predictionResults.append(result)

        print("finished predicting file "+str(n))
        n += 1

    #end prediction of all query articles

    #choose dominant result
    count = Counter(predictionResults)
    print("final prediction result:")
    print predictionResults
    print count

    count = Counter(expectedResults)
    print("final expected result:")
    print expectedResults
    print count

    matchCount = 0
    for i in range(len(predictionResults)):
        if predictionResults[i] == expectedResults[i]:
            matchCount += 1

    accuracy = float(matchCount) / len(predictionResults)

    print("accuracy = "+str(accuracy))


main()
