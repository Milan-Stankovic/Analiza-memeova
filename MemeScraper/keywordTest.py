import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
from gensim import corpora, models
from pprint import pprint
from itertools import chain
from gensim.summarization import keywords
import random
import csv
import nltk
from random import randint

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def read_Data(fname, arr):
    inputfile = csv.reader(open(fname, 'r', encoding="utf-8"))

    num =-1
    for idx, row in enumerate(inputfile):
        if idx %2 ==0:
            num =num+1
            arr.append(row[1]);

def write_Data(fname, content):
    text_file = open(fname, "a", encoding="utf-8")
    text_file.write(content+"\n")
    text_file.close()

def extractType(tokens, type, mode):
    retVal = []
    for idx in range(0, len(tokens)):
        if (tokens[idx][1][0] == type) == mode:
            temp = []
            temp.append(idx)
            temp.append(tokens[idx][0])
            retVal.append(temp)

    return retVal

def splitEntry(entry, splits, firstWordsIgnore):
    retVal = []
    frontSet = []
    backSet = []

    tokens = nltk.word_tokenize(text)
    posTokens = nltk.pos_tag(tokens)

    verbs = extractType(posTokens, 'V', True);
    nouns = extractType(posTokens, 'N', True);
    nonPunctuation = extractType(posTokens, '.', False);

    print("--------------------------------------------------------------------------------------------------------------")
    print("Verbs: ", verbs)
    print("Nouns: ", nouns)
    print("NonP: ", nonPunctuation)


    while splits>0 and (len(verbs)>0 or len(nouns)>0 or len(nonPunctuation)>0):
        print("Splits: ", splits, "LENS: v - ", len(verbs), " n - ", len(nouns), " nP - ", len(nonPunctuation))
        if len(verbs)>0:
            choose = randint(0, len(verbs)-1)
            print("BIRAM VERB: ", choose)
            if verbs[choose][0] >= firstWordsIgnore and verbs[choose][0]<len(verbs)-1:

                contentFirst = ""
                for i in range(0, verbs[choose][0]+1):
                    contentFirst = contentFirst + tokens[i] + " "

                contentSecond = ""
                for i in range(verbs[choose][0], len(tokens)):
                    contentSecond = contentSecond + tokens[i] + " "

                print(contentFirst + "--SPLIT--"+contentSecond)
                write_Data("trainFirst.csv", contentFirst)
                write_Data("trainSecond.csv", contentSecond)

                splits = splits - 1
            else:
                splits=splits
            del verbs[choose]
        elif len(nouns)>0:
            choose = randint(0, len(nouns)-1)
            print("BIRAM NOUN: ", choose)
            if nouns[choose][0] >= firstWordsIgnore and nouns[choose][0]<len(nouns)-1:

                contentFirst = ""
                for i in range(0, nouns[choose][0]+1):
                    contentFirst = contentFirst + tokens[i] + " "

                contentSecond = ""
                for i in range(nouns[choose][0], len(tokens)):
                    contentSecond = contentSecond + tokens[i] + " "

                print(contentFirst + "--SPLIT--" + contentSecond)
                write_Data("trainFirst.csv", contentFirst)
                write_Data("trainSecond.csv", contentSecond)

                splits = splits - 1
            else:
                splits = splits
            del nouns[choose]
        elif len(nonPunctuation)>0:
            choose = randint(0, len(nonPunctuation)-1)
            print("BIRAM NONPUN: ", choose)
            if nonPunctuation[choose][0] >= firstWordsIgnore and nonPunctuation[choose][0]<len(nonPunctuation)-1:

                contentFirst = ""
                for i in range(0, nonPunctuation[choose][0]+1):
                    contentFirst = contentFirst + tokens[i] + " "

                contentSecond = ""
                for i in range(nonPunctuation[choose][0], len(tokens)):
                    contentSecond = contentSecond + tokens[i] + " "

                print(contentFirst + "--SPLIT--" + contentSecond)
                write_Data("trainFirst.csv", contentFirst)
                write_Data("trainSecond.csv", contentSecond)

                splits = splits - 1
            else:
                splits = splits
            del nonPunctuation[choose]
    print("LENS: v - ", len(verbs), " n - ", len(nouns), " nP - ", len(nonPunctuation))

    print("Verbs: ", verbs)
    print("Nouns: ", nouns)
    print("NonP: ", nonPunctuation)
    print("--------------------------------------------------------------------------------------------------------------")


arr1=[]

read_Data('testData.csv', arr1)

for i in range(len(arr1)):
    text = arr1[i]
    splitEntry(text, 3, 4)

arr1=[]

read_Data('trainData.csv', arr1)

for i in range(len(arr1)):
    text = arr1[i]
    splitEntry(text, 3, 4)




