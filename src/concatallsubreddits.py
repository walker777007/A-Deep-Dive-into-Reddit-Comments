# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:41:31 2020

@author: walke
"""

import numpy as np
import pandas as pd
import glob
import timeit
#%%
all_subreddits = pd.read_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/allsubredditssmaller.csv')
all_subreddits = all_subreddits['subreddit']
all_subreddits = sorted(np.unique(all_subreddits),key=str.casefold)
#%%
for subreddit in all_subreddits:
    start_time = timeit.default_timer()
    csvs = glob.glob('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/reddit_comments/'+subreddit+'/*.csv')
    dfs=[]
    for line in csvs:
        df = pd.read_csv(line)
        dfs.append(df)
    final_df = pd.concat(dfs)
    
    small_df = final_df
    small_df = small_df.sort_values(by=['score'], ascending=False)
    small_df = small_df[:10000]
    small_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/allsubreddits_10000/'+subreddit+'.csv')
    print(subreddit,(timeit.default_timer() - start_time)/60,'minutes')
#%%
"""
final_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/allsubreddits/Drugs.csv')
#%%
smaller_df = final_df
smaller_df = smaller_df.sort_values(by=['score'], ascending=False)
smaller_df = smaller_df[:5000]
smaller_df.to_csv('C:/Users/walke/Documents/galvanize/capstones/A-Deep-Dive-into-Reddit-Comments/data/allsubredditssmaller/WTF.csv')
"""
