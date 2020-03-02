# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:14:10 2020

@author: walke
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import gensim 
import gensim.downloader as api
from gensim.models import Word2Vec 
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from MulticoreTSNE import MulticoreTSNE
import timeit
import pickle
import warnings
warnings.filterwarnings('ignore')
#%%
with open('data/doc2vec_feature_matrix.pkl', 'rb') as f:
    X = pickle.load(f)
#%%
tsne = TSNE(n_components=2,verbose=1, n_jobs=-1, random_state=1)
start_time = timeit.default_timer()
X_tsne = tsne.fit_transform(X)
print((timeit.default_timer() - start_time)/60,'minutes')
#%%
with open('data/tsne_matrix.pkl', 'wb') as f:
    pickle.dump(X_tsne, f)
#%%
#tsne = MulticoreTSNE(n_components=2,verbose=1, n_jobs=16, random_state=1)
#start_time = timeit.default_timer()
#X_tsne = tsne.fit_transform(X)
#print((timeit.default_timer() - start_time)/60,'minutes')
#%%
plt.scatter(X_tsne[:,0],X_tsne[:,1], alpha=0.01)
#%%
with open('data/tsne_matrix.pkl', 'wb') as f:
    pickle.dump(X_tsne, f)
#%%
print("Program is finished")
