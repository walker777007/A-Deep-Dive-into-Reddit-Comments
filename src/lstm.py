# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 14:21:26 2020

@author: walke
"""

import numpy as np
import pandas as pd
import pickle
import gensim 
import gensim.downloader as api
from gensim.models import Word2Vec 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import train_test_split
from denoise_text import denoise_text, stop_words
#%%
categories = pd.read_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/categories_small.csv')
#%%
categories = categories.drop(columns=['Unnamed: 0'])
#%%
#clean_text = categories['body'].map(denoise_text)
#%%
#with open('../data/cleaned_text_categories.pkl', 'wb') as f:
#    pickle.dump(clean_text, f)
#%%
data = pd.read_pickle('../data/cleaned_text_categories.pkl')
data = data.values
categories = categories['category'].values
#%%
X_train, X_test, y_train, y_test = train_test_split(data, categories, random_state=1)
#%%
with open('glove_twitter_200.pkl', 'rb') as f:
    glove_twitter = pickle.load(f)
#%%
train_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'mean') for z in X_train]))
test_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'mean') for z in X_test]))
#%%
with open('doc2vec_feature_matrix_train_categories.pkl', 'wb') as f:
    pickle.dump(train_vecs_glove_mean, f)
with open('doc2vec_feature_matrix_test_categories.pkl', 'wb') as f:
    pickle.dump(test_vecs_glove_mean, f)
#%%
with open('doc2vec_feature_matrix_train_categories.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('doc2vec_feature_matrix_test_categories.pkl', 'rb') as f:
    X_test = pickle.load(f)
#%%