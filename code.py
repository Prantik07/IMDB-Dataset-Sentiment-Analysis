import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

train_dataset = pd.read_csv('train.csv')
train_dataset = train_dataset.iloc[0:8000].values

def create_corpus(data) :
    corpus = []
    for i in range(0, data.shape[0]):
        review = data[i, 0]
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)
    return corpus
 
train_corpus = create_corpus(train_dataset)    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)
X = cv.fit_transform(train_corpus).toarray()
y = (train_dataset[:, -1]).astype(int)

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X, y) 

from sklearn.model_selection import cross_val_score
acc = cross_val_score(estimator = classifier, X = X, y = y, cv = 10, scoring = 'accuracy', n_jobs = -1)
acc.mean()

''' Testing on Test_dataset '''

test_dataset = pd.read_csv('test.csv')
test_dataset = test_dataset.iloc[:, :-1].values

test_corpus = create_corpus(test_dataset)
X_test = cv.transform(test_corpus).toarray()
y_pred = classifier.predict(X_test)
y_test = pd.read_csv('test.csv').iloc[:, -1].values

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) 
    

 
    
 
    
 