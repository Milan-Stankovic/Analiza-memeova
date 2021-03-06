import csv
import random
import heapq
from sklearn.model_selection import train_test_split


memeCount = 15

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

likeValues = []
maxValues = []
avgValues = []
faktorValues =[]

numberOfMemes= []

def normalize(list, numbers1, numbers2):
    for i, item in enumerate(list):
        factor = numbers1[i] / numbers2[i]

        for idx, value in enumerate(item):
            list[i][idx]= int(value*factor)


def saveFile(type, list):
    train_data, test_data = train_test_split(list, test_size=0.2, random_state=2324)
    with open('trainData.csv', 'a', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for item in train_data:
            writer.writerow([type,item])
    with open('testData.csv', 'a', encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        for item in test_data:
            writer.writerow([type, item])

def findAvgAndMax():
    for i in range(0, memeCount):
        avgValues.append(int(sum(likeValues[i])/len(likeValues[i])))
        maxValues.append(max(i for  i in likeValues[i]))
        faktorValues.append(int(maxValues[i]/avgValues[i]))
        numberOfMemes.append(len(likeValues[i]))

def grupByType(fname):

    inputfile = csv.reader(open(fname, 'r', encoding="utf-8"))

    for i in range(0, memeCount):
        likeValues.append(0)
        likeValues[i]=[]


    for idx, row in enumerate(inputfile):
        if idx %2 == 0 :

            if row[0] == 'ONE_DOES_NOT_SIMPLY':
                likeValues[0].append(int(row[3]))
                list_OneDoes.append(row[1])
            if row[0] == 'MOST_INTERESTING_MAN':
                likeValues[1].append(int(row[3]))
                list_MostInterest.append(row[1])
            if row[0] == 'SUCCESS_KID':
                likeValues[2].append(int(row[3]))
                list_Success_KID.append(row[1])
            if row[0] == 'BAD_LUCK_BRIAN':
                likeValues[3].append(int(row[3]))
                list_BadLuck.append(row[1])
            if row[0] == 'GOOD_GUY_GREG':
                likeValues[4].append(int(row[3]))
                list_GoodGuy.append(row[1])
            if row[0] == 'FOREVER_ALONE':
                likeValues[5].append(int(row[3]))
                list_ForeverAlone.append(row[1])
            if row[0] == 'ALL_THE_THINGS':
                likeValues[6].append(int(row[3]))
                list_AllTheThings.append(row[1])
            if row[0] == 'YO_DAWG':
                likeValues[7].append(int(row[3]))
                list_YoDawg.append(row[1])
            if row[0] == 'CONSPIRACY_KEANU':
                likeValues[8].append(int(row[3]))
                list_Keanau.append(row[1])
            if row[0] == 'WILLY_WONKA':
                likeValues[9].append(int(row[3]))
                list_Willy.append(row[1])
            if row[0] == 'WINTER_IS_COMING':
                likeValues[10].append(int(row[3]))
                list_Winter.append(row[1])
            if row[0] == 'FUTURAMA_FRY':
                likeValues[11].append(int(row[3]))
                list_Futurama.append(row[1])
            if row[0] == 'Y_U_NO':
                likeValues[12].append(int(row[3]))
                list_YUN.append(row[1])
            if row[0] == 'KERMIT_THE_FROG':
                likeValues[13].append(int(row[3]))
                list_Kermit.append(row[1])
            if row[0] == 'WHAT_IF_I_TELL_YOU':
                likeValues[14].append(int(row[3]))
                list_WhatIF.append(row[1])



grupByType('meme1sentiment.csv')

findAvgAndMax()

meme1_LikeValues = likeValues.copy()
meme1_AvgValues = avgValues.copy()
meme1_MaxValues = maxValues.copy()
meme1_FaktorValues = faktorValues.copy()
meme1_NumberOfMemes = numberOfMemes.copy()



likeValues.clear()
avgValues.clear()
maxValues.clear()
faktorValues.clear()
numberOfMemes.clear()


#SECOND MEME SITE



grupByType('meme2sentiment.csv')

findAvgAndMax()



#print(likeValues)
normalize(likeValues, meme1_AvgValues, avgValues)



allMeme_Number = []

allMeme_likeValues = []

for i in range(0, memeCount) :
    allMeme_Number.append( meme1_NumberOfMemes[i] + numberOfMemes[i])
    meme1_LikeValues[i].extend(likeValues[i])



#print(min(allMeme_Number))


#Uzimamo maximalno min broj meme-ova

indexes = []
for i in range(0, memeCount) :
    indexes.append(heapq.nlargest(min(allMeme_Number), range(len(meme1_LikeValues[i])), meme1_LikeValues[i].__getitem__))




list_OneDoes = [list_OneDoes[j] for j in indexes[0]]
list_MostInterest = [list_MostInterest[j] for j in indexes[1]]
list_Success_KID = [list_Success_KID[j] for j in indexes[2]]
list_BadLuck = [list_BadLuck[j] for j in indexes[3]]
list_GoodGuy = [list_GoodGuy[j] for j in indexes[4]]
list_ForeverAlone = [list_ForeverAlone[j] for j in indexes[5]]
list_AllTheThings = [list_AllTheThings[j] for j in indexes[6]]
list_YoDawg = [list_YoDawg[j] for j in indexes[7]]
list_Keanau = [list_Keanau[j] for j in indexes[8]]
list_Willy = [list_Willy[j] for j in indexes[9]]
list_Winter = [list_Winter[j] for j in indexes[10]]
list_Futurama = [list_Futurama[j] for j in indexes[11]]
list_YUN = [list_YUN[j] for j in indexes[12]]
list_Kermit = [list_Kermit[j] for j in indexes[13]]
list_WhatIF = [list_WhatIF[j] for j in indexes[14]]



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



#for idx, meme in enumerate(uniqueLables) :
 #   saveFile(meme, listAll[idx])





