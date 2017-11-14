import sys
import glob
import numpy as np

from random import randrange

from sklearn.neighbors import NearestNeighbors
from sklearn import svm
from sklearn import cross_validation


def createkNNData(k, seedArticles, sentimentData):
    neighbors = NearestNeighbors(n_neighbors = k, algorithm='ball_tree').fit(sentimentData)
    _, indices = neighbors.kneighbors(seedArticles)
    indices = reduce(np.union1d, indices).tolist()    
    return indices



def main():
    industry = sys.argv[1]
    k=0
    resultStr = ""


    #load sentiment data from corresponding industry
    sentimentData = []
    priceData = []
        
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

    # 10-fold cross validation using all data
    clf = svm.SVC()
    scores = cross_validation.cross_val_score(clf, sentimentData, priceData, cv=10)
    print("10-fold cross validation result of "+industry+" industry using all data:")
    print(scores)
    print(np.mean(scores))

    


    # 10-fold cross validation using kNN

    #random pick one as query sample
    sample = sentimentData[randrange(0, len(sentimentData))]

    query = []
    query.append(sample)
    query = np.array(query)

    kNNIndices = createkNNData(k, query, sentimentData)

    kNNData = []
    kNNPrice = []
    for index in kNNIndices:
        kNNData.append(sentimentData[index])
        kNNPrice.append(priceData[index])

    clf = svm.SVC()
    scores = cross_validation.cross_val_score(clf, kNNData, kNNPrice, cv=10)
    print("10-fold cross validation result of "+industry+" industry using kNN:")
    print(scores)
    print(np.mean(scores))






main()