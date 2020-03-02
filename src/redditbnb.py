# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:00:13 2020

@author: walke
"""

import matplotlib.style as style
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
import warnings
warnings.filterwarnings('ignore')
style.use('seaborn')
sns.set_style(style='darkgrid')
#%%
neolibs = glob.glob('C:/Users/walke/Desktop/reddit_comments/neoliberal/*.csv')
dfs=[]
for line in neolibs:
    df = pd.read_csv(line)
    dfs.append(df)
neolib = pd.concat(dfs)
chapos = glob.glob('C:/Users/walke/Desktop/reddit_comments/ChapoTrapHouse/*.csv')
dfs=[]
for line in chapos:
    df = pd.read_csv(line)
    dfs.append(df)
chapo = pd.concat(dfs)

df = pd.concat([neolib, chapo])
#%%
X = df['body'].values
y = df['subreddit'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X_train)
features = vectorizer.get_feature_names()
features = np.array(features)
#%%
bnb = BernoulliNB()
bnb.fit(X,y_train)