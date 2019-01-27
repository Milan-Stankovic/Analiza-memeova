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

mode =0
numberOfTypes =15
numberOfMemes = 1244


def read_corpus(fname):
    inputfile = csv.reader(open(fname, 'r', encoding="utf-8"))

    num =-1
    for idx, row in enumerate(inputfile):
        if idx %2 ==0:
            num =num+1
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[1]), [num])


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

print()

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

print(counts)
#print(maxs)
#print(ind)
print(origInd)


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



#prvi je meme drugi je koji je to cluster
paroviDict ={}

print(result)
print(toBeRemoved)

for i in range(0,numberOfTypes):

    yay = result[i].copy()

    for cluster in yay:
        meme = transposedInd[i].index(cluster)
      # print("Un : "+str(meme))
      # print("un : "+str(cluster))
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





print("-------------------------------------------")
print(paroviDict)
print(len(paroviDict))







'''''

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


#test = model.infer_vector(['Hide', 'the', 'pain','Grba','and', 'git', 'good'])
#result = model.docvecs.most_similar([test], topn=len(model.docvecs))

#print(result)
'''