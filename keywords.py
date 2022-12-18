from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import os
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

import datetime
from dateutil.relativedelta import relativedelta
from tqdm.notebook import tqdm_notebook

from collections import Counter
import sys

from scipy.stats import skew

nltk.download('stopwords')
nltk.download('punkt')

import string
stop = set(stopwords.words('english') + list(string.punctuation))
stop.update(['’','https','un','amp','``',"''","'s",'..','...',"n't",'--','”','–','//','“','like','also','put','ask','w/','unitednations'])

def clean_sentence(sentence):
    cleaned = [i for i in word_tokenize(sentence.lower()) if i not in stop]
    return np.array(cleaned)

df=pd.read_csv("./data_processed_for_patterns.csv")
words_List = np.array([clean_sentence(text) for text in tqdm_notebook(df['Text'])])


def get_top10_progression(condition, top10_words, alpha):

    top10_dyn = Counter(np.hstack(words_List[window_index_result[0] & condition]))
    top10_dyn = [np.array([top10_dyn.get(key,0) for key in top10_words])]
    
    for index in tqdm_notebook(window_index_result[1:]):
        dict_index = Counter(np.hstack(words_List[index & condition]))
        temp_vals = np.array([dict_index.get(key,0) for key in top10_words])
        top10_dyn.append(alpha*temp_vals + (1-alpha)*top10_dyn[-1])
        
    return(np.array(top10_dyn))

def get_top10_window(condition, min_times, alpha):
    top10_list = set()
    for window_pos in tqdm_notebook(window_index_result):
        top_words = Counter(np.hstack(words_List[window_pos & condition])).most_common(10)
        top_words = [tup[0] for tup in top_words if tup[1] >= min_times]
        top10_list.update(top_words)

    top10_list = list(dict.fromkeys(top10_list))
    top10_dyn = get_top10_progression(condition, top10_list, alpha)
    
    return(top10_list, top10_dyn)

min_times = 20
top10_pos, top10_dyn_pos = get_top10_window(df['Sentiment'] == 'Positive', min_times, 0.1)
top10_neg, top10_dyn_neg = get_top10_window(df['Sentiment'] == 'Negative', min_times, 0.1)
top10_neu, top10_dyn_neu = get_top10_window(df['Sentiment'] == 'Neutral', min_times, 0.1)
top10_tot, top10_dyn_tot = get_top10_window(1, min_times, 0.1)

