#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 00:06:01 2020

@author: Waquar Shamsi
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from textblob import TextBlob
import pickle

stock = 'WIPRO'
file_stock = 'Dataset/' + stock + '.csv'
stock_df = pd.read_csv(file_stock)
X = stock_df[['Date']]

train_dates = X.tail(210).head(180).values.tolist()
test_dates = X.tail(30).values.tolist()

date_count = 43801    #i.e, 2019-12-02


def get_subjectivity(text):
  return TextBlob(text).sentiment.subjectivity
def get_polarity(text):
  return TextBlob(text).sentiment.polarity

news_list = []

for stock_date in train_dates:
  year=stock_date[0][:4]
  month=stock_date[0][5:7]

  r = requests.get('https://economictimes.indiatimes.com/archivelist/year-'+str(year)+',month-'+str(month)+',starttime-'+str(date_count)+'.cms')
  date_count+=1
  soup = BeautifulSoup(r.content,"html.parser")

  archive_eco_news = soup.find_all ("ul",{"class":"content"})

  for dataCluster in archive_eco_news:
      for data in dataCluster.find_all("a"):
          content=TextBlob(data.text)
          content.lower()
          row=[]
          row.append(stock_date[0])
          row.append(str(content))
          news_list.append(row)

train_news = pd.DataFrame(news_list,columns=('Date','News'))
train_news['Subjectivity'] = train_news['News'].apply(get_subjectivity)
train_news['Polarity'] = train_news['News'].apply(get_polarity)

news_list = []

for stock_date in test_dates:
  year=stock_date[0][:4]
  month=stock_date[0][5:7]

  r = requests.get('https://economictimes.indiatimes.com/archivelist/year-'+str(year)+',month-'+str(month)+',starttime-'+str(date_count)+'.cms')
  date_count+=1
  soup = BeautifulSoup(r.content,"html.parser")

  archive_eco_news = soup.find_all ("ul",{"class":"content"})

  for dataCluster in archive_eco_news:
      for data in dataCluster.find_all("a"):
          content=TextBlob(data.text)
          content.lower()
          row=[]
          row.append(stock_date[0])
          row.append(str(content))
          news_list.append(row)

test_news = pd.DataFrame(news_list,columns=('Date','News'))
test_news['Subjectivity'] = test_news['News'].apply(get_subjectivity)
test_news['Polarity'] = test_news['News'].apply(get_polarity)


train_news_file = open('/content/drive/MyDrive/Colab Notebooks/train_news_data', 'wb') 
pickle.dump(train_news, train_news_file)

test_news_file = open('/content/drive/MyDrive/Colab Notebooks/test_news_data', 'wb') 
pickle.dump(test_news, test_news_file)

