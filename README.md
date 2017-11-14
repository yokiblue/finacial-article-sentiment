Project: analyze financial articles to predict stock price change

This solution adopts a topic-oriented method for sentiment analysis in financial news articles to offer daily predictability based on a supervised machine learning algorithm. 

Continuous historical market price information and cooperate financial news articles of companies categorized into different industries are used to establish the prediction of stock price movement placing on the sentimental polarity extracted from the news articles. 

To achieve higher information utilization and lower training cost, a semi-lazy mining paradigm (LAMP) is adopted in the implementation, where the training model is only computed and committed after receiving the prediction query and training data set is selected by applying KNN to the prediction query. 
