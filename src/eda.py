# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:36:02 2020

@author: walke
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
style.use('seaborn')
sns.set_style(style='darkgrid')
from nltk import word_tokenize
from denoise_text import denoise_text, stop_words
#%%
df = pd.read_csv('../data/allsubreddits_10000.csv')
df = df.drop(columns=['Unnamed: 0'])
AskWomen = df['body'][df['subreddit']=='AskWomen']
AskMen = df['body'][df['subreddit']=='AskMen']
femalefashionadvice = df['body'][df['subreddit']=='femalefashionadvice']
malefashionadvice = df['body'][df['subreddit']=='malefashionadvice']
#%%
askwomen_comment_length = np.mean(AskWomen.apply(word_tokenize).apply(len))
askmen_comment_length = np.mean(AskMen.apply(word_tokenize).apply(len))
femalefashionadvice_comment_length = np.mean(femalefashionadvice.apply(word_tokenize).apply(len))
malefashionadvice_comment_length = np.mean(malefashionadvice.apply(word_tokenize).apply(len))
#%%
fig, (ax1,ax2) = plt.subplots(1,2)
fig.add_subplot(111, frameon=False)
ax1.bar(['AskWomen','AskMen'],[askwomen_comment_length,askmen_comment_length])
ax2.bar(['femalefashionadvice','malefashionadvice'],[femalefashionadvice_comment_length,malefashionadvice_comment_length])
plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
plt.grid(b=None)
plt.ylabel('Average number of words per comment')
plt.xlabel('Subreddits')
plt.title('Comment Length on Gender Specific Subreddits')
plt.savefig('../plots/genderedcommentlength.png',dpi=640)