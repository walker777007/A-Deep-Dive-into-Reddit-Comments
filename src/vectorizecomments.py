# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:14:10 2020

@author: walke
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import timeit
import pickle
import warnings
warnings.filterwarnings('ignore')
#%%
def get_w2v_general(comment, size, vectors, aggregation='mean'):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    print(comment)
    for word in comment.split():
        try:
            vec += vectors[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if aggregation == 'mean':
        if count != 0:
            vec /= count
        return vec
    elif aggregation == 'sum':
        return vec
#%%
"""
glove_twitter = api.load("glove-twitter-200")

with open('glove_twitter_200.pkl', 'wb') as f:
    pickle.dump(glove_twitter, f)
    
google_news_300 = api.load("word2vec-google-news-300")

with open('google_news_300.pkl', 'wb') as f:
    pickle.dump(google_news_300, f)
"""
#%%
with open('glove_twitter_200.pkl', 'rb') as f:
    glove_twitter = pickle.load(f)
#%%
data = pd.read_pickle('../data/cleaned_text_lower_smaller.pkl')
data = data.values
#vectorizer = TfidfVectorizer(stop_words='english')
#X_vec = vectorizer.fit_transform(df['body'].values)
#%%
train_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'mean') for z in data]))
#%%
with open('doc2vec_feature_matrix_smaller.pkl', 'wb') as f:
    pickle.dump(train_vecs_glove_mean, f)
#%%
with open('doc2vec_feature_matrix_smaller.pkl', 'rb') as f:
    X = pickle.load(f)
#%%
X = X[0:-1:10]
#%%
tsne = TSNE(n_components=2,verbose=1, n_jobs=-1)
start_time = timeit.default_timer()
X_tsne = tsne.fit_transform(X)
print((timeit.default_timer() - start_time)/60,'minutes')
#%%
with open('tsne_matrix_smaller.pkl', 'wb') as f:
    pickle.dump(X_tsne, f)
#%%
df=pd.read_csv('../data/allsubredditssmaller.csv')
df = df['subreddit']
#%%
matplotlib.rcParams.update({'font.size': 8})
plt.scatter(X_tsne[:,0],X_tsne[:,1],alpha=0.01,zorder=0)
for i in sorted(np.unique(df),key=str.casefold):
    plt.scatter(np.mean(X_tsne[np.where(df==i)][:,0]),np.mean(X_tsne[np.where(df==i)][:,1]),color='k',zorder=0,alpha=0) 
    plt.annotate(i,xy=(np.mean(X_tsne[np.where(df==i)][:,0]), np.mean(X_tsne[np.where(df==i)][:,1])))    
#%%
kmeans = KMeans(n_clusters=50,random_state=1,n_jobs=-1)
start_time = timeit.default_timer()
kmeans.fit(X_tsne)
print((timeit.default_timer() - start_time)/60,'minutes')
#%%
for n_clusters in [50]:
    start_time = timeit.default_timer()
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1,2)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X_tsne) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=1)
    cluster_labels = clusterer.fit_predict(X_tsne)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X_tsne, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X_tsne, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], marker='.', alpha=0.035,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')
    
    print((timeit.default_timer() - start_time)/60,'minutes')