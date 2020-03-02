# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:13:41 2020

@author: walke
"""

import pandas as pd
import glob
#%%
csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/allsubreddits_10000/*.csv')
dfs=[]
for line in csvs:
    df = pd.read_csv(line)
    dfs.append(df)
final_df = pd.concat(dfs)
#%%
final_df = final_df.drop(columns=['Unnamed: 0'])
#%%
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/allsubreddits_10000.csv')
#%%
#csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/allsubreddits/*.csv')
#dfs=[]
#for line in csvs:
#    df = pd.read_csv(line)
#    dfs.append(df)
#final_df = pd.concat(dfs)
#%%
#final_df = final_df.drop(columns=['Unnamed: 0', 'Unnamed: 0.1'])
#%%
#final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/allsubreddits.csv')
#%%
csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/allsubredditssmaller/*.csv')
dfs=[]
for line in csvs:
    df = pd.read_csv(line)
    dfs.append(df)
final_df = pd.concat(dfs)
#%%
final_df = final_df.drop(columns=['Unnamed: 0'])
#%%
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/allsubredditssmaller.csv')