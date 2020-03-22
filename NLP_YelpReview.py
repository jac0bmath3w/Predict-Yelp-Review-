#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 21:19:03 2020
NLP for Yelp reviews
@author: jacob
"""

import nltk
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Preprocess
from sklearn.preprocessing import 

import string
from nltk.corpus import stopwords

# Vectorizing words

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# for keras models

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils


yelp = pd.read_csv('yelp.csv')
yelp['text length'] = yelp.text.apply(len)

g = sns.FacetGrid(yelp, col= 'stars')
g.map(plt.hist, 'text length', bins = 50)

sns.boxplot(data = yelp, x = 'stars', y = 'text length')

sns.countplot(data = yelp, x = 'stars')

stars = yelp.groupby(by = 'stars').mean()
sns.heatmap(stars.corr(), annot=True)

yelp_subsest = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]

X = yelp['text']
y = yelp['stars']

# Create a variable that is 1 when the number of stars is > 3 and 0 otherwise
y_cat = np.zeros(len(y))
y_cat[y > 3] = 1

labelEncoder_Y = LabelEncoder()
y_cat = labelEncoder_Y.fit_transform(y_cat)
#y = np_utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X,y_cat,test_size = 0.25)

def text_process(mess):
# =============================================================================
# Remove punctuation
# remove stop words
# return list of cleaned words
# =============================================================================
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    


#sms_data['vector'] = sms_data.message.apply(text_process)
bow_transformer = CountVectorizer(analyzer=text_process)
messages_bow = bow_transformer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
messages_tfidf = tfidf_transformer.fit_transform(messages_bow)

#tfidf_transformer.transform(bow_transformer.transform(X_test))


yelp_review = Sequential()
yelp_review.add(Dense(512,kernel_initializer = 'uniform', activation = 'relu', input_dim=messages_bow.shape[1]))
# model.add(Activation('relu'))
yelp_review.add(Dropout(0.1))
yelp_review.add(Dense(512,kernel_initializer = 'uniform', activation = 'relu'))
yelp_review.add(Dropout(0.5))
yelp_review.add(Dense(1, activation = 'sigmoid'))


yelp_review.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

history_yelp = yelp_review.fit(messages_tfidf, y_train, epochs=4, batch_size=32,verbose=1)
history_yelp_bow = yelp_review.fit(messages_bow, y_train, epochs=4, batch_size=32,verbose=1)


predictions = history_yelp.model.predict(tfidf_transformer.transform(bow_transformer.transform(X_test)))
predictions_bow = history_yelp_bow.model.predict(bow_transformer.transform(X_test))

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(messages_tfidf, y_train)
predictions_nb = nb.predict(tfidf_transformer.transform(bow_transformer.transform(X_test)))

nb_bow = MultinomialNB()
nb_bow.fit(messages_bow, y_train)
predictions_nb_bow = nb_bow.predict(bow_transformer.transform(X_test))

from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test, predictions)
confusion_matrix(y_test, predictions_bow > 0.5)
confusion_matrix(y_test, predictions_nb)
confusion_matrix(y_test, predictions_nb_bow)







# =============================================================================
# This part of the code is developing a neural network to determine the review from 1 to 5
# Since it was identified that bow gives better results than tfidf, we try that 
# =============================================================================

y = yelp['stars']


labelEncoder_Y = LabelEncoder()
y_all = labelEncoder_Y.fit_transform(y)
y_all = np_utils.to_categorical(y_all)

X_train, X_test, y_all_train, y_all_test = train_test_split(X,y_all,test_size = 0.25)

yelp_exact_review = Sequential()
yelp_exact_review.add(Dense(512,kernel_initializer = 'uniform', activation = 'relu', input_dim=messages_bow.shape[1]))
yelp_exact_review.add(Dropout(0.1))
yelp_exact_review.add(Dense(512,kernel_initializer = 'uniform', activation = 'relu'))
yelp_exact_review.add(Dropout(0.5))
yelp_exact_review.add(Dense(258,kernel_initializer = 'uniform', activation = 'relu'))
yelp_exact_review.add(Dropout(0.2))
yelp_exact_review.add(Dense(5, activation = 'softmax'))


yelp_exact_review.compile(loss='categorical_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

history_yelp_exact_bow = yelp_exact_review.fit(messages_bow, y_all_train, epochs=10, batch_size=32,verbose=1)

predictions_exact_bow = history_yelp_exact_bow.model.predict(bow_transformer.transform(X_test))

confusion_matrix(y_all_test.argmax(axis = 1), predictions_exact_bow.argmax(axis=1))
cr = classification_report(y_all_test.argmax(axis = 1), predictions_exact_bow.argmax(axis=1))

