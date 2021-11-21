# Simple Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json
import time
from scipy.sparse import data
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import SGDClassifier
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
#pass text column in the dataframe as X, and target_class as y
text=X
target_class=y
X_train, X_test, y_train, y_test = train_test_split(text, target_class, test_size=0.1, random_state=4)

#Put them in a pipeline, you can use different classification models
pipeline = Pipeline([
    ('vect', CountVectorizer(max_df=0.5, min_df = 5, ngram_range=(1,2), stop_words='english')),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(n_jobs=3))
])


parameters = {
    'vect__max_df':(0.2,0.5),
    'vect__min_df':(0.01,0.05),
    'tfidf__use_idf':(True, False),
    'tfidf__smooth_idf':(True, False),
    'clf__alpha': (0.0001,0.001),
    'clf__loss': ('hinge', 'log'), 
    'clf__penalty': ('l2', 'elasticnet'),
    'clf__max_iter': (100,200,500)
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected block
# find the best parameters for both the feature extraction and the classifier
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters)
    t0 = time.time()
    grid_search.fit(X_train, y_train)
    print("done in %0.3fs" % (time.time() - t0))
    print()
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
#Create a word2vec model with 64 dimensions

from gensim.models import Word2Vec

w2v_model1 = Word2Vec(data,
                #size is the number of dimensions of the N-dimensional space and Word2Vec maps the words into
                size=64, 
                window=10,
                min_count=1,
                workers=4)


w2v_model1.train(dataset, total_examples=w2v_model1.corpus_count, epochs=10)

wv = w2v_model1.wv


#Get the word vector for a given word
wv.most_similar('dataset')

#To be able to use in the pipeline, need to wrap into a dictionay
w2v = dict(zip(w2v_model1.wv.index2word, w2v_model1.wv.vectors))

#Averaging word vectors for all words in the text

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from pprint import pprint
from time import time
import logging

#Use pipeline 

rf =  Pipeline([('Word2Vec Vectorizer', MeanEmbeddingVectorizer(w2v)),
              ('Random Forest', RandomForestClassifier(n_estimators=50, criterion='entropy', verbose=True, n_jobs=3))])

svc = Pipeline([('Word2Vec Vectorizer', MeanEmbeddingVectorizer(w2v)),
              ('Support Vector Machine', SVC(kernel='rbf', C=0.5))])

sgd = Pipeline([('Word2Vec Vectorizer', MeanEmbeddingVectorizer(w2v)),
               ('Stochastic Gradient Descent', SGDClassifier(alpha=0.001, n_jobs=3))])
               
models = [('Random Forest', rf),
          ('Support Vector Machine', svc),
          ('Stochastic Gradient Descent', sgd)]

scores = [(name, cross_val_score(model, X_train, y_train, cv=2).mean()) for name, model, in models]
print(scores)

'''
This model predicts the salary of the employ based on experience using simple linear regression model.
'''
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the Test set results
y_pred = regressor.predict(X_test)
'''
# Saving model using pickle
#pickle.dump(regressor, open('model.pkl','wb'))
pickle.dump(models[0], open('model.pkl','wb'))
# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))
print(model.predict([[1.8]]))
