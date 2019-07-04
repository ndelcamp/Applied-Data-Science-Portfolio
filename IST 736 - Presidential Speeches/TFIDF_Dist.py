#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:10:26 2019

@author: drewhowell
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords 
from nltk.stem.snowball import SnowballStemmer
import os
import re
from datetime import datetime



path = 'speeches/'
files = []
pres = []
date = []
c = 0

# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        file = os.path.join(r, file)
        files.append(file)
        name = files[c].split('speeches/', 1)[-1]
        name = name.split('_', 1)[0]
        pres.append(name)
        groups = files[c].split('-')
        dat = '-'.join(groups[:3])
        dat = dat.split('_', 1)[-1]
        dat = datetime.strptime(dat, '%B-%d-%Y')
        date.append(dat)
        c +=1



#Put the contents of each file into list docs
docs = []   
    
for f in files:
    with open(f, 'r') as cur:
        docs.append(cur.read())

df = pd.DataFrame({'Text': docs})
df['pres'] = pres
df['date'] = date
#if file[-4:] == .txt:


# get rid of non-ascii characters
df['Text'] = df["Text"].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in x]))

# remove stop words
stop = set(stopwords.words('english')) 
df['noStop'] = df['Text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

#import re

head = '<div class=""view-transcript""><div class=""expandable-text""> <div class=""view-transcript-btn top""> <div class=""transcript-btn-inner""> <a class=""expandable-text-trigger"" data-close-label=""Hide Transcript"" data-open-label=""View Transcript"" href=""#dp-expandable-text"">View Transcript </a> </div> </div> <div class=""expandable-text-container"" id=""dp-expandable-text"" style=""display: none""> <div class=""transcript-inner""> <h3>Transcript</h3> <p>'
tail = '</p> </div> <div class=""view-transcript-btn bottom""> <a class=""expandable-text-trigger"" data-close-label=""Hide Transcript"" data-open-label=""View Transcript"" href=""#dp-expandable-text"">View Transcript </a> </div> </div> </div> </div>'
for row in range(0,len(df.iloc[1,])):
    df.iloc[row,1]= re.sub('<[^>]+>', '', df.iloc[row,1])

df['noStop'] = df['noStop'].str.replace('</p>', ' ').str.replace('<p>', ' ')
df['noStop'] = df['noStop'].str.replace('</li>', ' ').str.replace('<li>', ' ') \
                           .str.replace('<div class=""view-transcript""><h3>Transcript</h3><p>', '')\
                           .str.replace(',', ' ')\
                           .str.replace('.', ' ')\
                           .str.replace(';', ' ')\
                           .str.replace('<br/>','')\
                           .str.replace('</ul>','').str.replace('<ul>','')\
                           .str.replace('</a>','').str.replace('</div>','')\
                           .str.replace('href=""#dp-expandable-text"">View Transcript','')\
                           .str.replace('data-open-label=""View Transcript""','')\
                           .str.replace('data-close-label=""Hide Transcript""','')\
                           .str.replace('<div class=""view-transcript-btn bottom"">','')\
                           .str.replace('<a class=""expandable-text-trigger""','')\
                           .str.replace('View Transcript','')\
                           .str.replace('Transcript','')\
                           .str.replace('--',' ')\
                           .str.replace('-',' ')\
                           .str.replace('"','')\
                           .str.replace(':','')\
                           .str.replace('em','').str.replace('amp','')

                          # .str.replace(head,'').str.replace(tail,'')

from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import CountVectorizer

v = TfidfVectorizer()
sparse = v.fit_transform(df['noStop'])



from sklearn.cluster import KMeans  

# KMEANS 2 GROUPS
labeler = KMeans(n_clusters=2) 
labeler.fit(sparse.tocsr())  
lab2 = []
# print cluster assignments for each row 
for (row, label) in enumerate(labeler.labels_):   
  #print(row, label+1)
  lab2.append(label+1)
df['k2group'] = lab2

# KMEANS 3 GROUPS
labeler = KMeans(n_clusters=3) 
labeler.fit(sparse.tocsr())  
lab3 = []
# print cluster assignments for each row 
for (row, label) in enumerate(labeler.labels_):   
  #print(row, label+1)
  lab3.append(label+1)
df['k3group'] = lab3

# KMEANS 4 GROUPS
labeler = KMeans(n_clusters=4) 
labeler.fit(sparse.tocsr())  
lab4 = []
# print cluster assignments for each row 
for (row, label) in enumerate(labeler.labels_):   
  #print(row, label+1)
  lab4.append(label+1)
df['k4group'] = lab4

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

scores=[]
wordCount = []
AWL = []
for row in df['noStop']:
    score = analyzer.polarity_scores(row)
    scores.append(score)
    words = row.split()
    wordCount.append(len(words))
    AWL.append(len(row)/len(words))
    
df['SentiScore']=scores
df['negSore'] = [d['neg'] for d in scores]
df['posSore'] = [d['pos'] for d in scores]
df['wordCount'] = wordCount
df['AWL'] = AWL


df21 = df.loc[df['k2group']==1]
df22 = df.loc[df['k2group']==2]
df31 = df.loc[df['k3group']==1]
df32 = df.loc[df['k3group']==2]
df33 = df.loc[df['k3group']==3]
df41 = df.loc[df['k4group']==1]
df42 = df.loc[df['k4group']==2]
df43 = df.loc[df['k4group']==3]
df44 = df.loc[df['k4group']==4]

avDate = datetime(datetime.MINYEAR)
avDateList = []
var = [df21,df22,df31,df32,df33,df41,df42,df43,df44]
for g in var:
    for row in g['date']:
        avDate = avDate + row




#tokenize words by white space
df['words'] = df.noStop.str.strip().str.split('[\W_]+')
#df.head()

# stem words
stemmer = SnowballStemmer("english")
df['stemmed'] = df["words"].apply(lambda x: [stemmer.stem(y) for y in x])

# create Document ID column 0-227
df['Doc'] = np.arange(len(df))

# flatten the dataset into row = doc ID and single word
rows = []
for row in df[['Doc', 'words']].iterrows():
    r = row[1]
    for word in r.words:
        rows.append((r.Doc, word))

words = pd.DataFrame(rows, columns=['Doc', 'word'])
#words.head()

# change to lowercase
words['word'] = words.word.str.lower()
#words.head()

# get rid of blank cells in word column
words = words[words.word.str.len() > 0]
#words.head()


############
##### GETTING TF-IDF AND FREQ. DIST
###########

# get the counts of each word per document
counts = words.groupby('Doc')\
    .word.value_counts()\
    .to_frame()\
    .rename(columns={'word':'n_w'})
#counts.head()

# get number of words per document
word_sum = counts.groupby(level=0)\
    .sum()\
    .rename(columns={'n_w': 'n_d'})
#word_sum

# get TF 
# join counts to get frequencies
tf = counts.join(word_sum)
# get TF by dividing frequencies by total # of words in doc
tf['tf'] = tf.n_w/tf.n_d
#tf.head()

# get number of docs
c_d = words.Doc.nunique()
#c_d

# count DF per word
idf = words.groupby('word')\
    .Doc\
    .nunique()\
    .to_frame()\
    .rename(columns={'Doc':'i_d'})\
    .sort_values('i_d')
idf.head()

# get IDF per word
idf['idf'] = np.log(c_d/idf.i_d.values)
idf.head()

#join tf and idf
tfidf = tf.join(idf)
tfidf.head()

# calculate tf-idf for each word
tfidf['tf_idf'] = tfidf.tf * tfidf.idf
tfidf.head()

# plot top words overall
tfidf['tf_idf'].sort_values(ascending=False)[:25].plot(kind='bar')

# correlate top words with relative tweet length
#cortop = tf_idf.sort_values(by=['tf_idf'], ascending=False)
#cortop['n_d'][:5000].plot()
#cortop['n_d'].plot()











######################
########## END
######################

