import glob

from collections import Counter
from random import randrange
from sklearn.neighbors import NearestNeighbors
from functools import reduce
from datetime import timedelta
import datetime
import time
import sys
import os
import string
import multiprocessing
import operator
import csv
from multiprocessingMapreduce import simpleMapReduce

import pickle
import nltk

from nltk.corpus import sentiwordnet as swn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
from sklearn.neighbors import NearestNeighbors
from multiprocessingMapreduce import simpleMapReduce
import numpy as np



def outFiles(industry):
  n = 0
  while True:

        n += 1

        yield open("./trainingArt/"+industry+"/indArticle/%d.txt" % n, 'w')



def nonblankLines(f):

    for l in f:

        line = l.rstrip()

        if line:

            yield line



def findBetween(full, first, last):

    start = full.index(first)+len(first)

    end = full.index(last,start)


    return full[start:end]



# To split the article files into individual article

def splitArticle(fileList, industry):

    pat = 'Document LBA00000'    

    fs = outFiles(industry)

    outfile = next(fs)

    try:

        for fileIn in fileList:

            tmp = fileIn.split("/")

            company = tmp[len(tmp)-1].split(".")[0]
            outfile.write(company+"\n")

            with open(fileIn) as infile:

                for line in nonblankLines(infile):

                    if pat not in line:

                        outfile.write(line+" ")

                    else:                    

                        items = line.split(pat)

                        outfile.write(items[0])

                        for item in items[1:]:

                            outfile = next(fs)
                            outfile.write(company+"\n")

    except:

        print ("File cannot be splited.")



# To process the text of article   

def wordProcess(text):  

    tokenizer = RegexpTokenizer(r'\w+')

    snowb = stem.SnowballStemmer("english")

    

    # Tokenize 

    words = tokenizer.tokenize(text)

    # Remove stop words

    words = [w for w in words if w not in stopwords.words('english')]

    # Convert to lower case

    words = [w.lower() for w in words]

    # Stemming

    words = [snowb.stem(i) for i in words]

    return words



#################################################################################################################

###################################### added by ddemircioglu ####################################################

#################################################################################################################



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



# To extract the article content and save as [date, content]

def extractContent(fileIn):

    article = []

    pat2 = 'Released:'

    pat3 = '.000Z'

    content = ""

    company = ""


    with open(fileIn) as infile:

        lineNo = 1
        for line in infile:
            if lineNo == 1 :
                company += line
                lineNo += 1 
            else:
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

    

    article = [company,date,text]        

    return article




def stemmer(words):

    snowb = stem.SnowballStemmer("english")

    words = [snowb.stem(i) for i in words]

    return words



# To process the text of article   

def wordProcessWithoutStemming(text):  

    tokenizer = RegexpTokenizer(r'\w+')

    # Tokenize 

    words = tokenizer.tokenize(text)

    # Remove stop words & convert to lower case

    words = [w.lower() for w in words if w not in stopwords.words('english')]

    return words



# extract nouns from a single article

def extractNouns(article):

    taggedTokens = nltk.pos_tag(article)

    nouns = [word for word,pos in taggedTokens if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]

    return nouns    



# get union of all the kNN of input articles

def createkNNData(k, seedArticles, sentimentData):

    neighbors = NearestNeighbors(n_neighbors = k, algorithm='ball_tree').fit(sentimentData)

    _, indices = neighbors.kneighbors(seedArticles)

    indices = reduce(np.union1d, indices).tolist()    

    return indices



# generate N Xx numberofTopics matrix with random floats for using it with clustering

def dummyRandomData(N, numberOfTopics):    

    return np.random.rand(N, numberOfTopics)



def preprocessAllArticles(articleFileList):    

    processedArticles = list()

    nounCounter = Counter()

    nounList = list()    

    


    n=0

    for articleFile in articleFileList:

        articleContent = extractContent(articleFile)

        

        if not articleContent[0]:

            continue

        

        textToken = wordProcessWithoutStemming(articleContent[1])

        

        nounsInArticle = extractNouns(textToken)
        

        stemmedNouns = nounsInArticle

        nounCounter = nounCounter + Counter(stemmedNouns)                

                        

        n += 1

        if n == 30:

            break

    

    for n,c in nounCounter.items():

        if (c >= 10):

            nounList.append(n)
       

    return processedArticles, nounList






def fileToTopic(articleFile):
    output = []

    content = ''

    with open(articleFile, 'rt') as infile:
        next(infile)
        for line in infile:
            content += line

    text = content
    
    tokenizer = RegexpTokenizer(r'\w+')
    # Tokenize 
    words = tokenizer.tokenize(text)
    # Remove stop words & convert to lower case
    words = [w.lower() for w in words if w not in stopwords.words('english')]

    taggedTokens = nltk.pos_tag(words)
    nouns = [word for word,pos in taggedTokens if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS')]

    for word in nouns:
        output.append((word,1))

    return output


def countTopic(item):
    topic, occurances = item
    return (topic, sum(occurances))




#################################################################################################################

###################################### added by Jiayi ####################################################

#################################################################################################################        


def fileProcess(filename, article):  

    tokenizer = RegexpTokenizer(r'\w+')

    snowb = stem.SnowballStemmer("english")

    

    date = article[1]

    text = article[2]

    company = article[0]

    lines = text.split(".")



    content = company+str(date)+"\n"



    for line in lines:

        # Tokenize 

        words = tokenizer.tokenize(line)

        # Remove stop words

        words = [w for w in words if w not in stopwords.words('english')]

        # Convert to lower case

        words = [w.lower() for w in words]

        # Stemming

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


if __name__ == '__main__':
    industry = sys.argv[1]
    



    print('Remove old individual files')

    oldfiles1 = glob.glob('./trainingArt/'+industry+'/indArticle/*.txt')

    for f in oldfiles1:

        os.remove(f)

    oldfiles2 = glob.glob('./trainingArt/'+industry+'/processArt/*.txt')

    for f in oldfiles2:

        os.remove(f)

    oldfiles3 = glob.glob('./trainingArt/'+industry+'/artSentiment/*.txt')

    for f in oldfiles3:

        os.remove(f)


    print('Get original article list')


    fileList = glob.glob("./originalArticle/"+industry+"/*.txt")

 

    print('Split article bulks into individual articles')

    splitArticle(fileList, industry)


    print("original articles split into individual articles")

    #read indArticle files, process and store in required format (line by line)

    noOfArticles = 0
    try:

        articleFileList2 = glob.glob('./trainingArt/'+industry+'/indArticle/*.txt')
   

        for a in articleFileList2:


            article = extractContent(a)


            fileProcess(a, article)


            noOfArticles += 1

        print ("All Files are processed successfully!")

    except:

        print ("File cannot be processes!")


#-----Add by Yao ----
    # MapReduce Wordcount to find topic words
    print("start mapreduce")

    threshold = int(0.5 * float(noOfArticles))

    articleFileList3 = glob.glob('./trainingArt/'+industry+'/processArt/*.txt')
    mapper = simpleMapReduce(fileToTopic, countTopic)
    topicCounts = mapper(articleFileList3)
    topicCounts.sort(key = operator.itemgetter(1))
    topicCounts.reverse()
    topicListWithCount = [item for item in topicCounts if item[1]>threshold]
    topicList = [topicListWithCount[i][0] for i in range(len(topicListWithCount))]

    
#################################################################################################################

###################################### added by Jiayi ####################################################

#################################################################################################################        

        

    print "all topics found"


    # save all topics in txt

    topicStr = ""

    for i in range(len(topicList)):

        topicStr += str(topicList[i])+" "

    topicStr += "\n"



    with open("./trainingArt/"+industry+"/topics.txt", 'w') as outfile:

        outfile.write(topicStr) 

    print "all topics saved"


    print "start computing sentiments "



    #compute sentiment scores for each article
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

    try:

        topicList = []

        with open("./trainingArt/"+industry+"/topics.txt") as topicsFile:

            for line in topicsFile:

                topicList.extend(line.split())

            print "topicList: ", topicList

    except:

       print("topics read error")

    try:

        articleList = glob.glob('./trainingArt/'+industry+'/processArt/*.txt')

        n = 0

        for article in articleList:

            #a map in the form of (word, [sentiScore, count])
            try:
                sentiScores = {}
                priceFeature = [0.0, 0.0, 0.0]

                print "start article ", article
                company = ""
                date = ""
                label = 0.0
                with open(article) as infile:


                    lineNo = 1
                    for line in infile:

                        if lineNo ==1:
                            company = line.strip()

                            lineNo += 1
                            continue

                        if lineNo == 2:

                            date=line.strip()
                            
                            row = companyDict[company]
                            pastIndices, today, futureIndices = getColumnIndices(date, allDateList, priceFeatureNo, futurePriceNo)

                            label = getLabel(marketData, row, today, futureIndices)
                            priceFeature = getPriceChange(marketData, row, pastIndices, today)
                            lineNo += 1
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

                #save the sentiment values in txt


                articleSentiment = np.zeros(len(topicList))

                topicsFound = sentiScores.keys()

                for topic in topicsFound:

                    topicIdx = topicList.index(topic)

                    articleSentiment[topicIdx] = sentiScores[topic][0]

              
                articleSentiment = np.append(articleSentiment, priceFeature)
              

                n += 1

                filename = article.replace("processArt", "artSentiment")    

                with open(filename, 'w') as outfile:

                    outputStr = str(label)+"\n"

                    for v in articleSentiment:

                        outputStr += str(v)+" "



                    outfile.write(outputStr)

            except:
                continue    


            print("finished "+str(n)+" files")



    except:

        print("sentiment computation error")


    print "all sentiment scores computed"















