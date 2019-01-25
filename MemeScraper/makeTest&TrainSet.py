import csv
import random
from sklearn.model_selection import train_test_split
import numpy


list_OneDoes = []
list_MostInterest = []
list_Success_KID = []
list_BadLuck = []
list_GoodGuy = []
list_ForeverAlone = []
list_AllTheThings = []
list_YoDawg = []
list_Keanau = []
list_Willy = []
list_Winter = []
list_Futurama = []
list_YUN = []
list_Kermit = []
list_WhatIF = []


def saveFile(type, list):
    train_data, test_data = train_test_split(list, test_size=0.2, random_state=2324)
    with open('trainData.csv', 'a', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for item in train_data:
          #  print(item)
            writer.writerow([type,item])
    with open('testData.csv', 'a', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for item in test_data:
            writer.writerow([type, item])


def grupByType(fname):

    inputfile = csv.reader(open(fname, 'r', encoding="utf-8"))

    for idx, row in enumerate(inputfile):

        if idx %2 == 0 :
            if row[0] == 'ONE_DOES_NOT_SIMPLY':
                list_OneDoes.append(row[1]);
            if row[0] == 'MOST_INTERESTING_MAN':
                list_MostInterest.append(row[1]);
            if row[0] == 'SUCCESS_KID':
                list_Success_KID.append(row[1]);
            if row[0] == 'BAD_LUCK_BRIAN':
                list_BadLuck.append(row[1]);
            if row[0] == 'GOOD_GUY_GREG':
                list_GoodGuy.append(row[1]);
            if row[0] == 'FOREVER_ALONE':
                list_ForeverAlone.append(row[1]);
            if row[0] == 'ALL_THE_THINGS':
                list_AllTheThings.append(row[1]);
            if row[0] == 'YO_DAWG':
                list_YoDawg.append(row[1]);
            if row[0] == 'CONSPIRACY_KEANU':
                list_Keanau.append(row[1]);
            if row[0] == 'WILLY_WONKA':
                list_Willy.append(row[1]);
            if row[0] == 'WINTER_IS_COMING':
                list_Winter.append(row[1]);
            if row[0] == 'FUTURAMA_FRY':
                list_Futurama.append(row[1]);
            if row[0] == 'Y_U_NO':
                list_YUN.append(row[1]);
            if row[0] == 'KERMIT_THE_FROG':
                list_Kermit.append(row[1]);
            if row[0] == 'WHAT_IF_I_TELL_YOU':
                list_WhatIF.append(row[1]);



grupByType('meme1sentiment.csv')
grupByType('meme2sentiment.csv')

random.seed(2324)


listAll = []
listAll.append(list_OneDoes)
listAll.append(list_MostInterest)
listAll.append(list_Success_KID)
listAll.append(list_BadLuck)
listAll.append(list_GoodGuy)
listAll.append(list_ForeverAlone)
listAll.append(list_AllTheThings)
listAll.append(list_YoDawg)
listAll.append(list_Keanau)
listAll.append(list_Willy)
listAll.append(list_Winter)
listAll.append(list_Futurama)
listAll.append(list_YUN)
listAll.append(list_Kermit)
listAll.append(list_WhatIF)

brojac=-1
labels = []
with open('config.txt') as f:
    content = f.readlines()
    for line in content:
        brojac+=1
        if brojac!=0:
            line = line[:-1]
            parts = line.split(",")
            labels.append(parts[0])

uniqueLables=[ii for n,ii in enumerate(labels) if ii not in labels[:n]]



for idx, meme in enumerate(uniqueLables) :

    saveFile(meme, listAll[idx])





