import os
import numpy as np
import pandas as pd
import re
from tqdm import tqdm

# Libraries from Huggingface to download the transformer and classify in a simple way
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import csv
import urllib.request

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
 
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks: emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
# tokenizer = AutoTokenizer.from_pretrained("../twitter-roberta-base-sentiment")

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)
# model = AutoModelForSequenceClassification.from_pretrained("../twitter-roberta-base-sentiment")

def get_sentiment_tweet(text_input):
    text = preprocess(text_input)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return(scores)

def get_list_files(tweet_dir):
    list_files_raw = np.array(os.listdir(tweet_dir))
    list_files = np.sort(list_files_raw[['un_mentions_202' in file for file in list_files_raw]])
    return(list_files)


raw_data = pd.read_csv('./un_mentions_20220225-20220306.csv',lineterminator='\n')
list_tweets = raw_data[raw_data['Language'] == 'en'].reset_index(drop=True)
sentiment_list = pd.DataFrame([get_sentiment_tweet(tweet) for tweet in tqdm(list_tweets['Text'])])
list_tweets[['Negative','Neutral','Positive']] = sentiment_list
list_tweets.to_csv('./un_mentions_processed_20220225-20220326.csv',index=False)