# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:54:40 2020

@author: walke
"""

import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import nltk
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
import timeit
import pickle
import warnings
warnings.filterwarnings('ignore')
#%%
stop_words = set(stopwords.words('english')) 
stop_words.update({'aint','arent','cant','couldnt','didnt','doesnt','dont','hadnt',
                   'havent','mightnt','mustnt','neednt','shant','thatll','wasnt',
                   'werent','wont','wouldnt','youd','youll','youre','youve',"i'd",
                   "i'm","i'll","i've","can't","ain't","he'll","he'd","he's","she'll",
                   "she'd","she's","it'll","we'll","we'd","we've","we're","they'll",
                   "they'd","they're","they've"})
stop_words.update({word.capitalize() for word in stop_words})
tknzr = TweetTokenizer()
from nltk.tokenize import TweetTokenizer
# strips <> html tags from reviews
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Define function for removing special characters like '/' in 'don/'t'
def remove_special_characters(text):
    pattern=r'[^a-zA-z0-9\.\'\’\-\$\%\s](?!((\d*\.)?\d+))(?!\b(?:[A-Z][a-z]*){1,})'
    text=re.sub(pattern,' ',text)
    pattern=r'[^a-zA-z0-9\'\’\-\$\%\s](?!((\d*\.)?\d+))(?!\b(?:[A-Z][a-z]*){1,})'
    text=re.sub(pattern,' ',text)
    return text

#Stemming the text
def simple_lemmatizer(text):
    wn = WordNetLemmatizer()
    text= ' '.join([wn.lemmatize(word) for word in text.split()])
    return text
# remove apostrophe from contractions
def strip_contractions(text):
    pattern=r'[^a-zA-z0-9\.\-\$\%\s](?!((\d*\.)?\d+))(?!\b(?:[A-Z][a-z]*){1,})'
    text=re.sub(pattern,'',text)
    return text
# cleaning the text
def denoise_text(text):
    text = strip_html(text)
    text = text.replace('&#x200B;','')
    text = text.replace("’","'")
    text = text.replace("\'","'")
    text = text.replace('^',' ')
    text = text.replace('\n',' ')
    text = text.replace('\\','') 
    text = text.replace('“',' ')
    text = text.replace('”',' ')
    text = remove_special_characters(text)
    text = text.replace('-',' ')
    text = text.replace("'s",' ')
    #text = simple_lemmatizer(text)
    #text = strip_contractions(text)
    text = re.sub(' +',' ',text)
    text = re.sub(r'[~`!@#&*()+=_{}[\]|:;\\"<>?]','',text)
    text = re.sub(r'[0-9]','',text) #removing digits for doc2vec NOT tfidf
    text = tknzr.tokenize(text)
    text = [w for w in text if not w in stop_words]
    text = ' '.join(text)
    text = text.lower()
    print(text)
    return text
#%%
df = pd.read_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/allsubreddits_10000.csv')
df = df.drop(columns=['Unnamed: 0'])
#%%
start_time = timeit.default_timer()
clean_text = df['body'].map(denoise_text)
print((timeit.default_timer() - start_time)/60,'minutes')

with open('../data/cleaned_text_10000.pkl', 'wb') as f:
    pickle.dump(clean_text, f)
