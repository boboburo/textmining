# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Text mining: Finding hidden information in CRM data
# 
# ## Supporting Project for Text Mining Presentation 
# 
# Author: Jon Sedar  
# Date: Spring 2014  
# 
# ### Data notes
# 
# Obviously the client data is confidential, so I've created a dummy dataset based on a list of enforced NAMA properties which has similar features to the client dataset.
# 
# Available as a Google Doc here:   
# https://docs.google.com/spreadsheet/ccc?key=0AjOXYk-Wh9M2dGt1TWxjZzc5dDBtY29NMjRvUDhUTXc

# <codecell>

# -- coding: utf-8 --
#    @filename: text_mining_presentation_support.py
#    @author: jon.sedar@applied.ai
#    @copyright: Applied AI Ltd UK 2014
#    @license: BSD-3

## Libraries and global options
%matplotlib inline
#%matplotlib qt
%qtconsole --colors=linux 
from __future__ import division, print_function
import os
import sys
import re
import string
from time import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

from nltk.tokenize.punkt import PunktWordTokenizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from sklearn.cluster import MiniBatchKMeans
from sklearn.cross_validation import ShuffleSplit
from sklearn.neighbors import RadiusNeighborsClassifier

from IPython.display import Image
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# pandas formatting
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.mpl_style', 'default')
pd.set_option('display.notebook_repr_html', True)
pd.set_option('display.expand_frame_repr', True)
pd.set_option('display.precision', 5)
pd.set_option('display.max_colwidth', 50)

# handle temporary deprection warning in pandas for describe()
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", UserWarning)
    

# <codecell>

## local custom functions
def rstr(df):
    """Return an imitation of R's str() and summary() functions"""
    return pd.concat([df.describe().transpose()
                      ,pd.DataFrame(df.dtypes)
                      ,df.head().transpose()],axis=1)

def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

rex1 = re.compile(r"nan")
rex2 = re.compile(r"not defined")
rex3 = re.compile(r"[0-9A-Za-z]")
rex4 = re.compile(r"dublin\s[0-9]")
rex5 = re.compile(r"^[0-9 ]+$")

def clean_text(x):
    """Clean up single raw text string"""
    x1 = x.strip()                      # strip left and right
    x2 = ' '.join(x1.split())           # replace all whitespace with single space
    x3 = removeNonAscii(x2)             # remove non-ascii
    x4 = x3.lower()                     # lowercase
    x5 = rex1.sub("",x4)                # regex remove "nan"
    x6 = rex2.sub("",x5)                # regex remove "not defined"
    
    # split and replace space between dublin and number    
    if rex4.search(x6) != None:
        x0 = ""
        for word in x6.split():
            if word == "dublin":
                x0 = x0 + word + "_"
            else:
                x0 = x0 + word + " "
        x6 = x0.strip()
        
    x7 = rex5.sub("",x6)                # regex remove unbound number    

    return(x7)


def clean_tokens(x):
    """Clean loose punctuation from token lists and concat into single spaced string"""    
    idxs = [i for i, item in enumerate(x) if rex3.search(item)]
    conc = " ".join([x[i] for i in idxs])
    return(conc)

# <markdowncell>

# ## Data Import and Cleaning

# <codecell>

## Import raw data and quick look
raw = pd.read_csv('data/dummy_data_nama_july2011.csv')
raw.rename(columns=lambda x: '_'.join(x.lower().split()), inplace=True)  # colnames: remove spaces and lower
raw.rename(columns=lambda x: x.translate(string.maketrans("",""), string.punctuation), inplace=True)  # colnames: remove punct

print(raw.shape)
rstr(raw)

# <codecell>

## Select interesting features and concat into new column for ease of use
feats = ['address1','address2','address3','country','fulladdress'
         ,'assetdescription','additionalassetdescription','receiversfirm']
df = raw[feats]
df['concat'] = df.apply(lambda row: ' '.join([str(col) for col in row]),axis=1)

for i in range(0,10):
    print(df.ix[i,'concat'])

# <codecell>

## Clean and Tokenise preserving some punctuation
df['concat_clean'] = df['concat'].apply(clean_text)
df['token'] = df['concat_clean'].apply(PunktWordTokenizer().tokenize)
df['token_clean'] = df['token'].apply(clean_tokens)
for i in range(0,10):
    print(df.ix[i,'token_clean'])

# <markdowncell>

# ## Vectorization (TF-IDF)
# Transforming the text into a numeric representation

# <codecell>

## Create TF-IDF Vectors
vectorizer = TfidfVectorizer(max_df=0.95, max_features=20000,ngram_range=(1,2)
                            ,stop_words=None #'english'
                            ,use_idf=True,smooth_idf=True)
X = vectorizer.fit_transform(df["token_clean"])
vectorizer.get_feature_names()
X.shape

# <markdowncell>

# ## Dimensionality Reduction (SVD)
# Transforming the sparse matrix into a dense matrix representation
# 
# Using Singular Value Decomposition (SVD) a method akin to Prinicipal Components Analysis to map the full space onto a reduced feature space

# <codecell>

## Dimensionality reduction and re-normalization
lsa = TruncatedSVD(n_components=100)
Y = lsa.fit_transform(X)
Z = Normalizer(copy=False).fit_transform(Y)
Z.shape
Z[:10,:10]

# <markdowncell>

# ## Initial Clustering (K-Means)
# Split up the dataset into more manageable chunks

# <codecell>

## Find good number of clusters to use
inertia = {}

for i in range(1,21):
    km = MiniBatchKMeans(n_clusters=i,init='k-means++',n_init=10,random_state=0
                         ,init_size=1000,batch_size=1000,verbose=False)
    t0 = time()
    km.fit(Z)
    print("testing n= "+str(i)+", done %0.3fs" % (time() - t0))
    inertia[i] = km.inertia_   

#%% Choose best num clusters
min(inertia.iterkeys(), key=(lambda key: inertia[key]))    
        
# fig = plt.figure(1, figsize=(6, 6))
# plt.clf()
plt.scatter(inertia.keys(), inertia.values()
            ,c=[1.0-j for j in inertia.values()],cmap="YlGnBu"
            ,alpha=0.8,marker='o',linewidths=0.3)

f = interpolate.interp1d(inertia.keys(), inertia.values())
xnew = np.arange(min(inertia.keys()),max(inertia.keys()),0.1)
plt.plot(xnew,f(xnew),color="b",linestyle="--")
plt.show    

# <codecell>

## Run using fixed chosen cluster count
clust = 15
km = MiniBatchKMeans(n_clusters=clust,init='k-means++',n_init=10,random_state=0
                         ,init_size=1000,batch_size=1000,verbose=False)
km.fit(Z)

#%% Make quick observations of clusters
A = pd.DataFrame(Z)
A["label1"] = km.labels_
A["label1_color"] = (km.labels_+1)/(max(km.labels_)+1)
A.groupby("label1")[0].count()


## Quick plot
fig = plt.figure(1, figsize=(10, 10))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
labels = A["label1"]
rndpts = np.sort(np.random.choice(A.shape[0], min(1000,A.shape[0]), replace=False))

ax.scatter(A.iloc[rndpts,0],A.iloc[rndpts,1],A.iloc[rndpts,2]
        ,c=A.iloc[rndpts].label1_color
        ,cmap="jet",alpha=0.8,marker='o',linewidths=0.1)

# plot centroids
cntrs = km.cluster_centers_[:,0:3]
ax.scatter(cntrs[:,0],cntrs[:,1],cntrs[:,2]
        ,c=np.arange(1,clust+1,1)/(clust+1)
        ,cmap="jet",alpha=1,marker='D',s=100, linewidths=1.5)

ax.set_xlabel('SVD 1st')
ax.set_ylabel('SVD 2nd')
ax.set_zlabel('SVD 3rd')
plt.show()

# <markdowncell>

# ## Extract clusters of companies for manual labelling

# <codecell>

df['label'] = km.labels_
df.insert(0, 'human_suggested_id', "")

for i in np.unique(km.labels_):
    df.loc[df.label==i].to_csv('data/human_tagging/df_lbl_'+str(i)+'.csv')

# <codecell>

## Take a look at an example set
df.loc[df.label==0]    

