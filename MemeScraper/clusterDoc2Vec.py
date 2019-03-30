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


def read_Test(fname):
    inputfile = csv.reader(open(fname, 'r', encoding="utf-8"))

    num =-1
    for idx, row in enumerate(inputfile):
        if idx %2 ==0:
            num =num+1
            yield gensim.utils.simple_preprocess(row[1])


def read_corpus(fname):
    inputfile = csv.reader(open(fname, 'r', encoding="utf-8"))

    num =-1
    for idx, row in enumerate(inputfile):
        if idx %2 ==0:
            num =num+1
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[1]), [num])





mode = 0
numberOfTypes = 15
numberOfMemes = 1244
numberOfTestMemes = 311

text = list(read_corpus('trainData.csv'))
#text.extend(list(read_corpus('memegenerator2.csv')))

model = gensim.models.doc2vec.Doc2Vec(vector_size=300, dm=0, min_count=2,epochs=60, iter=60, min_cont=2, workers=2, seed=2324, alpha = 0.025, min_alpha=0.00025)

model.build_vocab(text)

#%time


if mode == 1 :
    print("Training starting ")
    start = time.time()
    model.train(text, total_examples=model.corpus_count, epochs=model.iter)
    end = time.time()
    model.save('doc2VecWeights0')
   # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    print("Training finished and saved after: ")
    print(end - start)
else :
    model = Doc2Vec.load('doc2VecWeights0')




#Po Train data :  1244, idu redom, svaki je od po jednog meme-a
#Test data : 311

kmeans_model = KMeans(n_clusters=15, init='k-means++', max_iter=500, n_init=25, precompute_distances=True, n_jobs=-1, algorithm="auto")

if mode==1 :
    print("Training Kmeans starting ")
    start = time.time()
    #X = kmeans_model.fit(model.docvecs.vectors_docs)
    l = kmeans_model.fit_predict(model.docvecs.vectors_docs)
    end = time.time()
    print("Training Kmeans finished and saved after: ")
    print(end - start)

    pickle.dump(kmeans_model, open('kmeansModel0','wb'))
else :
    kmeans_model=pickle.load(open('kmeansModel0', 'rb'))
labels=kmeans_model.labels_.tolist()

num =-1

vals =[]

for i in range(0,numberOfTypes):
    vals.append([])

#print()

for idx,item in enumerate(labels):
    if idx%numberOfMemes ==0 :
        num=num+1
    vals[num].append(item)


counts = []

for i in range(0,numberOfTypes):
    counts.append([])
   # print("Meme : "+ str(i))
    for j in range(0, numberOfTypes):
       # print(vals[i].count(j))
        counts[i].append(vals[i].count(j))


maxs = []
ind = []

origInd = []

for i in range(0,numberOfTypes):
    maxs.append(max(counts[i]))
    ind.append(counts[i].index(max(counts[i])))
    origInd.append(numpy.argsort(counts[i])[::-1][:numberOfTypes])

#print(counts)

for i in range(len(counts)):
    print(counts[i])
#print(maxs)
#print(ind)
#print(origInd)


#New kul function
transposedInd = []
result=[]
toBeRemoved = []
whenRemoved = []
for i in range(0,numberOfTypes):
    transposedInd.append([])
    result.append([])
    toBeRemoved.append([])
    for j in range(0, numberOfTypes):
        transposedInd[i].append(origInd[j][i])
    for x in transposedInd[i]:
        if x not in result[i]:
            result[i].append(x)
        else :
            if x not in toBeRemoved[i]:
                toBeRemoved[i].append(x)
    for y in toBeRemoved[i] :
       result[i].remove(y)

#print(transposedInd)

#prvi je meme drugi je koji je to cluster
paroviDict ={}

#print(result)
#print(toBeRemoved)

unique = result

for i in range(0,numberOfTypes):

   # print("Ulazak u glavni for : " + str(i))
   # print("UNIEQUE SU : " + str(unique[i]))
    tempUn = unique[i].copy()
    for cluster in tempUn:        #sve unique odmah dodajem
       # print("TRAZIM UNIQUE : " +str(cluster))
        for f in range(0, numberOfTypes):   #nije transponovano pa prolazim ovako kroz sve nizove i gledam
           # print("F JE : " + str(f))
           # print( "ORIGINAL IND JE :" + str(origInd[f]))
          #  print("ORIGINAL IND NA I JE :" + str(origInd[f][i]))
            if origInd[f][i]==cluster :
                if f not in paroviDict :
                 #   print("DODAJEM UNIQE : "+ str(f)+" vrednost clustera : " + str(cluster))
                    paroviDict[f] = cluster #meme numb = cluster num
                    for j in range(0, numberOfTypes): #Izbacujem dodat cluster iz unique
                        if cluster in unique[j]:
                            unique[j].remove(cluster)
                        if cluster in toBeRemoved[j]:
                            toBeRemoved[j].remove(cluster)
                        if cluster in origInd[j] :
                            [-1 if x==cluster else x for x in origInd[j]] # menjam uzete vrednosti na -1
                    break


    tempRemove = toBeRemoved[i].copy()
   # print("TO BE REMOVED SU : " + str(toBeRemoved[i]))
    for conflictingCluster in tempRemove :
        #print("KONFLIKTNI CLUSTER JE : "+str(conflictingCluster))
        indexes = []
        indexes.clear()

        conflictingMaxValues = []
        conflictingMaxValues.clear()
        for h in range(0, numberOfTypes):   #nije transponovano pa prolazim ovako kroz sve nizove i gledam
            if origInd[h][i] == conflictingCluster: # trazim indekse
                if h not in paroviDict :
                    indexes.append(h)
                    conflictingMaxValues.append(counts[h][conflictingCluster]) #same vrednosti
       # print("U konfliktu su max vrednosti : " + str(conflictingMaxValues))
        if len(conflictingMaxValues)> 0:
            index = conflictingMaxValues.index(max(conflictingMaxValues))
          #  print("DODAJEM KONFLITNAN : " + str(indexes[index]) + " vrednost clustera : " + str(conflictingCluster))
            paroviDict[indexes[index]] = conflictingCluster

            for j in range(0, numberOfTypes):  # Izbacujem dodat cluster iz unique
                if conflictingCluster in unique[j]:
                    unique[j].remove(conflictingCluster)
                if conflictingCluster in toBeRemoved[j]:
                    toBeRemoved[j].remove(conflictingCluster)
                if conflictingCluster in origInd[j]:
                    [-1 if x == conflictingCluster else x for x in origInd[j]]  # menjam uzete vrednosti na -1

  #  print("ZA SADA : " + str(paroviDict))


#print("------------------------------")
#print(counts)

#meme je key cluster je value

success = []

for key, value in sorted(paroviDict.items()) :
    #print(key)
    success.append(counts[key][value]*100/numberOfMemes);

print(sum(success)/len(success))
for idx, rate in enumerate(success) :
    print(str(idx) +". meme success rate is : " + str(rate) + " %")


'''''

for i in range(0,numberOfTypes):

    yay = result[i].copy()

    for cluster in yay:
        meme = transposedInd[i].index(cluster)
        #print("Un : "+str(meme))
        #print("un : "+str(cluster))
       #paroviDict[meme] = cluster
        paroviDict[cluster] = meme
        #print("Dodao u : " + str(cluster) + " broj " + str(meme))
        for j in range(0,numberOfTypes):
            if cluster in result[j]:
                result[j].remove(cluster)
            if cluster in toBeRemoved[j]:
                toBeRemoved[j].remove(cluster)
            if cluster in transposedInd[j]:
                transposedInd[j].remove(cluster)


    pls = toBeRemoved[i].copy()
    for conflictingCluster in pls:
        indices = [z for z, x in enumerate(transposedInd[i]) if x == conflictingCluster]  #sada moram od transposovati -.-
        temp = []
        temp.clear()
       # print("CONFLICT : ")
        #print(conflictingCluster)
        for row in indices:
           temp.append(counts[row][conflictingCluster])
        #winner winner chicken diner
        winRow = temp.index(max(temp))
        gudRow = indices[winRow]
       # print("INDEXI : ")
       # print(indices)
        #print(gudRow)
        #print(conflictingCluster)
        #paroviDict[gudRow] = conflictingCluster
        print("Dodao u : " + str(conflictingCluster) + " broj " + str(gudRow))
        paroviDict[conflictingCluster] = gudRow
        for h in range(0, numberOfTypes):
            if conflictingCluster in result[h]:
                result[h].remove(conflictingCluster)
            if conflictingCluster in toBeRemoved[h]:
                toBeRemoved[h].remove(conflictingCluster)


'''''


#print("-------------------------------------------")
#print(paroviDict)
#print(len(paroviDict))

# meme key cluster value







pca = PCA(n_components=2).fit(model.docvecs.vectors_docs)
datapoint = pca.transform(model.docvecs.vectors_docs)


plt.figure





label1 = ["#FF0000","#00FF00","#0000FF","#FFFF00","#00FFFF","#FF00FF","#FF1493","#808080","#800000","#808000","#008000","#800080","#008080", "#000080", "#800000"]
color = [label1[i] for i in labels]
plt.scatter(datapoint[:, 0], datapoint[:, 1], c=color)

centroids = kmeans_model.cluster_centers_
centroidpoint = pca.transform(centroids)
plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')
plt.show()


testText = list(read_Test('testData.csv'))

#print(len(testText))

test = []



for idx, txt in enumerate(testText):
    test.append(model.infer_vector(txt))

predictedClusters =kmeans_model.predict(test)
#print(predictedClusters)


testCount = []

num =-1

for i in range(0, len(testText)):

    if i%numberOfTestMemes == 0:
        testCount.append(0)
        num=num+1
    if predictedClusters[i]== paroviDict[num]:
        testCount[num] = testCount[num]+1

#print(testCount)

#print("TEST")
#print("---------------------------------------------------")
#print("TEST")

successTest = []
print(sum(testCount)*100/numberOfTestMemes)
for i in range(0, numberOfTypes):
    successTest.append(testCount[i]*100/numberOfTestMemes)
    print(str(i) + ". Test meme success rate is : " + str(successTest[i]) + " %")








#test = model.infer_vector(['Hide', 'the', 'pain','Grba','and', 'git', 'good'])
#result = model.docvecs.most_similar([test], topn=len(model.docvecs))

#print(result)
