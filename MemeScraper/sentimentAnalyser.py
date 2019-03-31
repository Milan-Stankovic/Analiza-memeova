import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv
import os
import collections
import smart_open
import random

with open('memegenerator2.csv', 'r', encoding="utf-8") as f:
  reader = csv.reader(f)
  your_list = list(reader)

sid = SentimentIntensityAnalyzer()

for idx, row in enumerate(your_list):
    if idx%2==0:
        ss = sid.polarity_scores(row[1])
        for k in ss:
            row.append(ss[k])

with open("meme2sentiment.csv", 'a', encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    for idx, item in enumerate(your_list):
        if idx % 2 == 0:
            writer.writerow(item)
            #print(item)