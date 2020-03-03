<p align="center">
<img src="images/reddit_header.png" width="851" height="315">
</p>

# A Deep Dive into Reddit Comments
**Clustering reddit comments and subreddits**
<br>Walker Stevens
\
[Linkedin](https://www.linkedin.com/in/walker-stevens-31783087/) | [Github](https://github.com/walker777007)
\
[Slides](https://docs.google.com/presentation/d/1QkEkJUW1XqevWUOdSR_jPeH1tglSzEwxV0EEx10kcIc/edit?usp=sharing)

## Table of Contents

* [Motivation](#motivation)
* [Data Exploration](#data-exploration)
  * [Pipeline](#pipeline)
* [Emotional Sentiment Analysis](#emotional-sentiment-analysis)  
* [Topic Modeling](#modeling)
  * [GloVe Word2Vec](#glove-word2vec)
  * [T-distributed Stochastic Neighbor Embedding](#t-distributed-stochastic-neighbor-embedding)
  * [K-Means Clustering](#k-means-clustering)
* [Conclusion](#conclusion)

## Motivation

I've used reddit over the years, and I was curious how some of the more popular subreddits were related to each other.  Reddit(https://reddit.com) covers a lot of topics, from sports, video games, food, film, and much more.  My idea is to represent all of these topics by collecting reddit comments and clustering them together so I could essentially show the common themes and topics people are talking about, and the common ties between certain subreddits.

## Data exploration

### Pipeline

Where I got the data:
* [Google Big Query](https://bigquery.cloud.google.com/dataset/fh-bigquery:reddit_comments)

I queried the reddit_comments database on Google Big Query, specifically looking at comments between January and September of 2019.  Querying only long form comments (over 100 characters and excluding links and other forms of unhelpful punctuation), I picked the top comments from 101 active subreddits, which represent a fairly diverse set of topics.  The entire list of subreddits I picked can be seen [here]().  

Once all the indivudal CSV files were queried using SQL, I used pandas in order to group them into dataframes, and proceeded to do all my calculations and tests after.  I originally had over 10 million comments in total, but due to computation time, I had to cut it down to a total of 1,010,000 comments (10,000 per subreddit).  Even with a million comments, computation time was still pretty steep, so I used an AWS EC2 instance for the more heavy calculations.

Ignoring stop words, the corpus contained 253,725 unique words and each comment had an average length of ~ 27 words.

The most common words can be seen below.  They tend to be generic, albeit non-stopwords.  As we will see later, these words don't have nearly the impact in clustering topics as other specific words will:
<p align="center">
<img src="plots/wordcloud.png" width="525" height="350">
</p>

One interesting I had noticed in the data, was the discrepancy in the comment lengths between gender based subreddits.  
<p align="center">
<img src="plots/genderedcommentlength.png" width="800" height="550">
</p>

## Emotional Sentiment Analysis

Using the [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm), I thought it would be interesting to see how the emotional sentiments of certain subreddits compared to each other.  The way the NRC Emotion Lexicon works is that certain words are scored as anger, anticipation, disgust, fear, joy, sadness, surprise, trust and positive, negative.  The way in which I applied this was by essentially parsing each word in the comment and scoring each emotion as the number of emotional words divided by the total amount of words in the comment.  Here we can see the top 10 subreddits for the emotions of anger, joy (I call it happy), and sadness.
<p align="center">
<img src="plots/angrysubreddits.png" width="800" height="382">
</p>
<p align="center">
<img src="plots/happysubreddits.png" width="800" height="382">
</p>
<p align="center">
<img src="plots/sadsubreddits.png" width="800" height="382">
</p>
