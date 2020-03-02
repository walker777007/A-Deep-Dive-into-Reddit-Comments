# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:15:19 2020

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
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer 
from tqdm import tqdm_notebook as tqdm
from tqdm import trange
from sklearn.manifold import TSNE
import timeit
import pickle
from denoise_text import denoise_text, stop_words
#%%

def text_emotion(df, column):
    '''
    Takes a DataFrame and a specified column of text and adds 10 columns to the
    DataFrame for each of the 10 emotions in the NRC Emotion Lexicon, with each
    column containing the value of the text in that emotions
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with ten new columns
    '''

    new_df = df.copy()

    filepath = ('C:/Users/walke/Desktop/NRC-Sentiment-Emotion-Lexicons/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt')
    emolex_df = pd.read_csv(filepath,
                            names=["word", "emotion", "association"],
                            sep='\t')
    emolex_words = emolex_df.pivot(index='word',
                                   columns='emotion',
                                   values='association').reset_index()
    emotions = emolex_words.columns.drop('word')
    emo_df = pd.DataFrame(0, index=df.index, columns=emotions)
    
    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()
    
    with tqdm(total=len(list(new_df.iterrows()))) as pbar:
        start_time = timeit.default_timer()
        for i, row in new_df.iterrows():
            pbar.update(1)
            document = word_tokenize(new_df.loc[i][column])
            for word in document:
                word = stemmer.stem(word.lower())
                emo_score = emolex_words[emolex_words.word == word]
                if not emo_score.empty:
                    for emotion in list(emotions):
                        emo_df.at[i, emotion] += emo_score[emotion]
            if i in np.ceil(np.linspace(101000,1009999,10)).astype(np.int):
                print(i/1010000,'finished')
                print((timeit.default_timer() - start_time)/60,'minutes')

    new_df = pd.concat([new_df, emo_df], axis=1)

    return new_df
#%%
df = pd.read_csv('../data/allsubreddits_10000.csv')
df = df.drop(columns=['Unnamed: 0'])
df['body'] = df['body'].map(denoise_text)
#%%
start_time = timeit.default_timer()
hp_df = text_emotion(df, 'body')
print((timeit.default_timer() - start_time)/60,'minutes')
#%%
with open('../data/emotions_df.pkl', 'wb') as f:
    pickle.dump(hp_df, f)
#%%
hp_df['word_count'] = hp_df['body'].apply(word_tokenize).apply(len)
emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']
for emotion in emotions:
    hp_df[emotion] = hp_df[emotion] / hp_df['word_count']
    
hp_df = hp_df.fillna(0)

emotion_arr = hp_df[['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative',
       'positive', 'sadness', 'surprise', 'trust']].values

emotion_dict={}

for i in sorted(np.unique(hp_df['subreddit']),key=str.casefold):
    emotion_dict[i] =  np.mean(emotion_arr[np.where(hp_df['subreddit']==i)],axis=0)
#%%    
with open('../data/emotions_dictionary.pkl', 'wb') as f:
    pickle.dump(emotion_dict, f)
#%%
with open('../data/emotions_dictionary.pkl', 'rb') as f:
    emotion_dict = pickle.load(f)
#%%
subreddits=[]
anger=[]
for subreddit, values in emotion_dict.items():
    subreddits.append(subreddit)
    anger.append(values[0])
subreddits = np.array(subreddits)
anger = np.array(anger)

fig, ax = plt.subplots()

ax.bar(subreddits[np.argsort(anger)[::-1]][:10],100*anger[np.argsort(anger)[::-1]][:10], color='r')
ax.set_title('Top 10 Angriest Subreddits')
ax.set_xlabel('Subreddit')
ax.set_ylabel('Percentage of Comment Containing Angry Words')
plt.tight_layout()
plt.savefig('../plots/angrysubreddits.png',dpi=640)
#%%
subreddits=[]
joy=[]
for subreddit, values in emotion_dict.items():
    subreddits.append(subreddit)
    joy.append(values[4])
subreddits = np.array(subreddits)
joy = np.array(joy)

fig, ax = plt.subplots()

ax.bar(subreddits[np.argsort(joy)[::-1]][:10],100*joy[np.argsort(joy)[::-1]][:10], color='yellow')
ax.set_title('Top 10 Happiest Subreddits')
ax.set_xlabel('Subreddit')
ax.set_ylabel('Percentage of Comment Containing Happy Words')
plt.tight_layout()
plt.savefig('../plots/happysubreddits.png',dpi=640)
#%%
subreddits=[]
sadness=[]
subreddits = np.array(subreddits)
sadness = np.array(sadness)

fig, ax = plt.subplots()

ax.bar(subreddits[np.argsort(sadness)[::-1]][:10],100*sadness[np.argsort(sadness)[::-1]][:10], color='blue')
ax.set_title('Top 10 Saddest Subreddits')
ax.set_xlabel('Subreddit')
ax.set_ylabel('Percentage of Comment Containing Sad Words')
plt.tight_layout()
plt.savefig('../plots/sadsubreddits.png',dpi=640)
#%%
tsne = TSNE(random_state=1,n_jobs=-1)
X_tsne = tsne.fit_transform(emotions)
#%%
matplotlib.rcParams.update({'font.size': 8})
for i in subreddits:
    plt.scatter(X_tsne[np.where(subreddits==i)][:,0],X_tsne[np.where(subreddits==i)][:,1],color='k',zorder=0,alpha=0) 
    plt.annotate(i,xy=(X_tsne[np.where(subreddits==i)][:,0], X_tsne[np.where(subreddits==i)][:,1]))
plt.title('Sentiment Map')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.tight_layout()
plt.savefig('../plots/sentimentmap.png',dpi=640)