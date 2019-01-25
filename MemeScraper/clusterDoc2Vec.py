import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
warnings.filterwarnings(action='ignore', category=DeprecationWarning, module='sklearn')
import gensim
import os
import collections
import smart_open
import random
import csv
from sklearn.cluster import KMeans
from sklearn import metrics
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def read_corpus(fname):
    inputfile = csv.reader(open(fname, 'r', encoding="utf-8"))

    for idx, row in enumerate(inputfile):
        if idx %2 ==0:
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[1]), [idx])


text = list(read_corpus('memegenerator1.csv'))
text.extend(list(read_corpus('memegenerator2.csv')))

model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=2, epochs=40, workers=7)

model.build_vocab(text)

#%time
model.train(text, total_examples=model.corpus_count, epochs=model.epochs)


kmeans_model = KMeans(n_clusters=15, init='k-means++', max_iter=100)
X = kmeans_model.fit(model.docvecs.vectors_docs)
labels=kmeans_model.labels_.tolist()


l = kmeans_model.fit_predict(model.docvecs.vectors_docs)
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