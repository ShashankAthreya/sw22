import os
import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt

import datetime
from dateutil.relativedelta import relativedelta
from tqdm.notebook import tqdm_notebook

from scipy.stats import skew

def get_skewness(sentiment):
    return str(np.round(skew(repeat_timescales[repeat_timescales['Sentiment'] == sentiment]['Daily']),2))


def simplify_dates(datasetT):
    datasetT['Datetime'] = [datetime.datetime.strptime(re.sub('\+00:00','',i), date_format) for i in datasetT['Datetime']]
    return datasetT
    
def round_sentiments(datasetT):
    datasetT[['Negative','Neutral','Positive']] = np.rint(datasetT[['Negative','Neutral','Positive']])
    return datasetT

def get_cleaned_df(file_read):
    dataset = pd.read_csv(file_read,lineterminator='\n')
    dataset = simplify_dates(dataset)
    dataset = round_sentiments(dataset)
    dataset = dataset.iloc[::-1].reset_index(drop=True)
    return dataset

def tidy_data(df):
    df['Text'] = [re.sub('https.*','', line) for line in df['Text']]
    df = df.drop_duplicates(subset=['Text']).reset_index(drop=True)
    sentiment_column = np.full(len(df),'        ')
    sentiment_column[df['Negative'] >= 0.51] = 'Negative'
    sentiment_column[df['Positive'] >= 0.51] = 'Positive'
    sentiment_column[sentiment_column == '        '] = 'Neutral'
    df.insert(6, "Sentiment", sentiment_column, True)
    return(df)


# Extract just the time and convert to hours
def get_day_counts(input_set):
    timeList = [timestamp.time() for timestamp in input_set]
    list_in_day = np.array([(t.hour * 60 + t.minute) * 60 + t.second for t in timeList])/3600
    return list_in_day

# Convert to day of the week and put on a range from 0 to 7 depending on day and time.
def get_week_counts(input_set):
    dayList = np.array([timestamp.weekday() for timestamp in input_set])
    list_in_week = dayList + get_day_counts(input_set)/24
    return list_in_week

df= pd.read_csv("./un_mentions_processed_20221202-20221203.csv")
df = tidy_data(df)
df['Datetime'] = pd.to_datetime(df['Datetime'], errors='coerce')

# Get how many tweets happen in a week or day, to find patterns
list_all_day = get_day_counts(df['Datetime'])
list_all_week = get_week_counts(df['Datetime'])

# Group into a single dataframe for convenience
repeat_timescales = pd.concat([pd.DataFrame(list_all_day,columns=['Daily']),
                               pd.DataFrame(list_all_week,columns=['Weekly']),
                               df['Sentiment']],axis=1)



sns.set(rc = {'figure.figsize':(16,8)})
ax = sns.histplot(data=repeat_timescales, x='Daily', hue='Sentiment', stat="density", element="step")
ax.set_xlim([0,24])
ax.set(xlabel='Time of the day', title='Distribution of tweets during the day')
plt.savefig("./Distribution of tweets during the day.png")

print('Neutral skewness: '+get_skewness('Neutral'))
print('Positive skewness: '+get_skewness('Positive'))
print('Negative skewness :'+get_skewness('Negative'))

sns.set(rc = {'figure.figsize':(16,8)})
wPlot = sns.histplot(data=repeat_timescales, x='Weekly', hue='Sentiment', stat="density", element="step",)
wPlot.set_xticks(np.arange(0.5,7.5,1))
wPlot.set_xticklabels(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']);
wPlot.set_xlim([0,7]);
wPlot.set(title='Distribution of tweets during the week');
plt.savefig("./Distribution of tweets during the week.png")

df.to_csv("./data_processed_for_patterns.csv")
