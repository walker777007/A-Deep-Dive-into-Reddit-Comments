# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 13:03:37 2020

@author: walke
"""

import pandas as pd
import glob
#%%
csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/animals_small/*.csv')
dfs=[]
for line in csvs:
    df = pd.read_csv(line)
    dfs.append(df)
final_df = pd.concat(dfs)
#%%
final_df = final_df.drop(columns=['Unnamed: 0'])
final_df['category'] = 'animals'
#%%
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/categories_small/animals_small.csv')
#%%
csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/finance_small/*.csv')
dfs=[]
for line in csvs:
    df = pd.read_csv(line)
    dfs.append(df)
final_df = pd.concat(dfs)
#%%
final_df = final_df.drop(columns=['Unnamed: 0'])
final_df['category'] = 'finance'
#%%
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/categories_small/finance_small.csv')
#%%
csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/memes_small/*.csv')
dfs=[]
for line in csvs:
    df = pd.read_csv(line)
    dfs.append(df)
final_df = pd.concat(dfs)
#%%
final_df = final_df.drop(columns=['Unnamed: 0'])
final_df['category'] = 'memes'
#%%
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/categories_small/memes_small.csv')
#%%
csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/movies-tv_small/*.csv')
dfs=[]
for line in csvs:
    df = pd.read_csv(line)
    dfs.append(df)
final_df = pd.concat(dfs)
#%%
final_df = final_df.drop(columns=['Unnamed: 0'])
final_df['category'] = 'movies/tv'
#%%
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/categories_small/movies-tv_small.csv')
#%%
csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/politics_small/*.csv')
dfs=[]
for line in csvs:
    df = pd.read_csv(line)
    dfs.append(df)
final_df = pd.concat(dfs)
#%%
final_df = final_df.drop(columns=['Unnamed: 0'])
final_df['category'] = 'politics'
#%%
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/categories_small/politics_small.csv')
#%%
csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/relationships-romance_small/*.csv')
dfs=[]
for line in csvs:
    df = pd.read_csv(line)
    dfs.append(df)
final_df = pd.concat(dfs)
#%%
final_df = final_df.drop(columns=['Unnamed: 0'])
final_df['category'] = 'relationship/romance'
#%%
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/categories_small/relationships-romance_small.csv')
#%%
csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/science-math_small/*.csv')
dfs=[]
for line in csvs:
    df = pd.read_csv(line)
    dfs.append(df)
final_df = pd.concat(dfs)
#%%
final_df = final_df.drop(columns=['Unnamed: 0'])
final_df['category'] = 'science/math'
#%%
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/categories_small/science-math_small.csv')
#%%
csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/sports_small/*.csv')
dfs=[]
for line in csvs:
    df = pd.read_csv(line)
    dfs.append(df)
final_df = pd.concat(dfs)
#%%
final_df = final_df.drop(columns=['Unnamed: 0'])
final_df['category'] = 'sports'
#%%
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/categories_small/sports_small.csv')
#%%
csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/videogames_small/*.csv')
dfs=[]
for line in csvs:
    df = pd.read_csv(line)
    dfs.append(df)
final_df = pd.concat(dfs)
#%%
final_df = final_df.drop(columns=['Unnamed: 0'])
final_df['category'] = 'videogames'
#%%
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/categories_small/videogames_small.csv')
#%%
csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/categories_small/*.csv')
dfs=[]
for line in csvs:
    df = pd.read_csv(line)
    dfs.append(df)
final_df = pd.concat(dfs)
#%%
final_df = final_df.drop(columns=['Unnamed: 0'])
#%%
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/categories_small.csv')

