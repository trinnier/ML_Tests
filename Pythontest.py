import requests
import json
import pandas as pd
from pprint import pprint
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import requests
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

#class Sentiment:
#    NEGATIVE = 'NEGATIVE'
#    NETURAL = 'NETURAL'
#    POSITIVE = 'POSITIVE'


#class Review:
#    def __init__(self, text, score):
#        self.text = text
#        self.score = score
#        self.sentiment = self.get_sentiment()

#    def get_sentiment(self):
#        if self.score <= 2:
#            return Sentiment.NEGATIVE
#        elif self.score == 3:
#            return Sentiment.NETURAL
#        else: #score 4 or 5
#            return Sentiment.POSITIVE

#file_name = 'Books_small.json'
#reviews = []
#with open(file_name) as f:
#    for line in f:
#        review = json.loads(line)
#        print(review['reviewText'])
#        print(review['overall'])
#        reviews.append(Review(review['reviewText'], review['overall']))

#training, test = train_test_split(reviews, test_size=0.33, random_state=42 )
#print(len(training))

train_x = [x.text for x in training]
train_y = [y.text for y in training]

test_x = [x.text for x in test]
test_y = [y.text for y in test]

vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vectors, train_y)
print(clf_svm.predict(test_x_vectors[0]))
#print(predictions)

#print(clf_svm.score(test_x_vectors, test_y))


### Accuray and F1 score 

#clf_svm.predict(test_x_vectors[0])
#clf_svm.predict(test_x_vectors[0])

#print(train_x[0])
#print(train_x_vectors[0].toarray())


#print(reviews[5].sentiment)

##clf_dec = DecisionTreeClassifier()
##clf_dec.fit(train_x_vectors, train_y)
##clf_dec.predict(test_x_vectors[0])
#print(y)
