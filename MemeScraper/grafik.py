import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='sklearn')
import gensim
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os
import collections
import smart_open
import random
import csv
import time
from sklearn.cluster import KMeans
from sklearn import metrics
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from gensim.test.utils import get_tmpfile
import scipy
import numpy
#scipy.show_config()
import matplotlib.pyplot as plt

def read_Data(fname, arr):
    inputfile = csv.reader(open(fname, 'r', encoding="utf-8"))

    num =-1
    for idx, row in enumerate(inputfile):
        if idx %2 ==0:
            num =num+1
            arr.append([row[0], row[3], row[4], row[5], row[6], row[7], row[8]]);
                                        #       neg2     neu3     pos4     compound5
            #print([row[3], row[5], row[6]])
def read_DataRelevant(fname, arr, noRelevant, column):#column=3 => upvotes column=4 => coments
    inputfile = csv.reader(open(fname, 'r', encoding="utf-8"))

    num =-1
    for idx, row in enumerate(inputfile):
        if idx %2 ==0:
            num =num+1
            if int(row[column]) >= noRelevant:
                arr.append([row[0], row[3], row[4], row[5], row[6], row[7], row[8]]);
                                        #       neg2     neu3     pos4     compound5
            #print([row[3], row[5], row[6]])
def skrnavstina(arr):
    for i in range(len(arr)):
        for j in range(len(arr)):
            if int(arr[i][1]) > int(arr[j][1]):
                arr[i], arr[j] = arr[j], arr[i]

def izvuciKolonu(arr, l):
    temp=[]
    for i in range(len(arr)):
        temp.append(arr[i][l])
    return temp
def izvuciTip(arr, tip, kolonaZaPlotovanje, bul):
    temp=[]
    for i in range(len(arr)):
        if arr[i][0]==tip and (bul or float(arr[i][kolonaZaPlotovanje]) > 0.25):
           # print(arr[i])
            temp.append(arr[i])
    return temp
def graph(dataSet, tipovi, ignorisiNule, sentimentPart):
    for i in range(len(tipovi)):
        print("Graphing for: " + tipovi[i])
        arr = izvuciTip(dataSet, tipovi[i], sentimentPart, ignorisiNule)
        skrnavstina(arr)
        printArr(arr)
        x=izvuciKolonu(arr, 1)
        y=izvuciKolonu(arr, sentimentPart)
        x=convert(x)
        y=convert(y)
        #print(x)
        #print(y)
        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.bar(x, y)
        axes = plt.gca()
        ax.set_title(tipovi[i].replace('_', ' '), fontweight='bold')
        ax.set_ylabel('Sentiment')
        ax.set_xlabel('Number of likes')
        #axes.set_ylim([max(y), min(y)])
        #axes.set_xlim([min(x), max(x)])
        plt.show()

def graphBoxPlot(dataSet, tipovi, ignorisiNule, sentimentPart):
    for i in range(len(tipovi)):
        print("Graphing for: " + tipovi[i])
        arr = izvuciTip(dataSet, tipovi[i], sentimentPart, ignorisiNule)
        skrnavstina(arr)
       # printArr(arr)
        #x=izvuciKolonu(arr, 1)
        y=izvuciKolonu(arr, sentimentPart)
        #x=convert(x)
        y=convert(y)
        #print(x)
        #print(y)

        fig=plt.figure()

        ax = fig.add_subplot(111)
        ax.boxplot(y)
        ax.set_title(tipovi[i].replace('_', ' '), fontweight='bold')
        ax.set_ylabel('Sentiment')
        #axes = plt.gca()
        #axes.set_ylim([max(y), min(y)])
        #axes.set_xlim([min(x), max(x)])
        plt.show()

def convert(arr):
    ret=[]
    for i in range(len(arr)):
        ret.append(float(arr[i]))
    return ret

def printArr(arr):
    for i in range(len(arr)):
        print(arr[i])


def graphByInterval(dataSet, tipovi, ignorisiNule):
    print()


def getMemeSentCount(dataSet, sent):#sent 0-neg 1-neu 2-pos 3-compound
    ret=0
    for i in range(len(dataSet)):
        print(str(dataSet[i])+'---'+str(getMaxArrIdx(dataSet[i])))
        if getMaxArrIdx(dataSet[i])==sent:
            ret=ret+1
    return ret


def getMemeSentCount2(dataSet, sent):#sent 0-neg 1-neu 2-pos 3-compound
    ret=0
    for i in range(len(dataSet)):
        print(str(dataSet[i])+'---'+str(getMaxArrIdx2(dataSet[i])))
        if getMaxArrIdx2(dataSet[i])==sent:
            ret=ret+1
    return ret

def getMaxArrIdx2(arr):
    max=3;
    for i in range(3,len(arr)-1):
        if i==4:
            continue
        if float(arr[i]) > float(arr[max]):
            max=i
    if arr[3]==0 and arr[5]==0:
        return -1
    return max-3

def getMaxArrIdx(arr):
    max=3;
    for i in range(3,len(arr)-1):
        if float(arr[i]) > float(arr[max]):
            max=i
    return max-3

def graphMemeSentCountWithNeutral(wholeDS, types):
    for i in range(len([0,1,2])):
        count=[]
        for j in range(len(types)):
            count.append(getMemeSentCount(izvuciTip(wholeDS, types[j], -1, True), i))
        print('Ploting bar graph for sent:'+str(i))
        plt.bar(types, count)
        plt.show()


def graphMemeSentCount(wholeDS, types):
    for i in range(len([0,1,2])):
        count=[]
        for j in range(len(types)):
            count.append(getMemeSentCount2(izvuciTip(wholeDS, types[j], -1, True), i))
        print('Ploting bar graph for sent:'+str(i))
        plt.bar(types, count)
        plt.show()
arr1=[]
tips=[
    'ONE_DOES_NOT_SIMPLY',
    'MOST_INTERESTING_MAN',
    'SUCCESS_KID',
    'BAD_LUCK_BRIAN',
    'GOOD_GUY_GREG',
    'FOREVER_ALONE',
    'ALL_THE_THINGS',
    'YO_DAWG',
    'CONSPIRACY_KEANU',
    'WILLY_WONKA',
    'WINTER_IS_COMING',
    'FUTURAMA_FRY',
    'Y_U_NO',
    'KERMIT_THE_FROG',
    'WHAT_IF_I_TELL_YOU'
]

'''


'''
read_Data('meme2sentiment.csv', arr1)
#read_DataRelevant('meme1sentiment.csv', arr1, 100, 3)
graph(arr1, tips, True, 5)#3neg 5pos

#graphBoxPlot(arr1, tips, False, 5)#3neg 5pos
#graphMemeSentCount(arr1, tips)
#graphMemeSentCountWithNeutral(arr1, tips)