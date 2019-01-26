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
#scipy.show_config()

mode =0


def read_corpus(fname):
    inputfile = csv.reader(open(fname, 'r', encoding="utf-8"))

    num =-1
    for idx, row in enumerate(inputfile):
        if idx %2 ==0:
            num =num+1
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[1]), [num])


text = list(read_corpus('trainData.csv'))
#text.extend(list(read_corpus('memegenerator2.csv')))

model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2,epochs=60, iter=60, min_cont=2, workers=2, seed=2324, alpha = 0.025, min_alpha=0.00025)

model.build_vocab(text)

#%time


if mode == 1 :
    print("Training starting ")
    start = time.time()
    model.train(text, total_examples=model.corpus_count, epochs=model.iter)
    end = time.time()
    model.save('doc2VecWeights')
   # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
    print("Training finished and saved after: ")
    print(end - start)
else :
    model = Doc2Vec.load('doc2VecWeights')




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

    pickle.dump(kmeans_model, open('kmeansModel','wb'))
else :
    kmeans_model=pickle.load(open('kmeansModel', 'rb'))
labels=kmeans_model.labels_.tolist()



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

