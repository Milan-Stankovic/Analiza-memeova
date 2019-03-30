import pandas as pd
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer as stemmer
from nltk.stem.porter import *
import numpy as np
import nltk
from gensim import corpora, models
from pprint import pprint
from itertools import chain

stemmer = stemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

processed_docs=[]
data = pd.read_csv('meme1sentiment.csv', error_bad_lines=False)

data_text = data[['TEXT']]
data_text['index'] = data_text.index
documents = data_text

print(documents)
np.random.seed(2018)
nltk.download('wordnet')
for i in range(len(documents)):
    doc_sample = documents[documents['index'] == i].values[0][0]
    words = []
    print("Procesing meme" + str(i))
    for word in doc_sample.split(' '):
        words.append(word)
    temp=preprocess(doc_sample)
    processed_docs.append(temp)

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.filter_extremes(no_below=10, no_above=1, keep_n=100000)#ovde parametre fitujemo, obavezno u rad
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]

print("LDA training...please wait, might take a while")
lda_model = gensim.models.LdaModel(bow_corpus, num_topics=15, id2word=dictionary, passes=2, minimum_probability=0)
lda_model.save('ldaTest')
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} Words: {}'.format(idx, topic))

maxr=584
minr=0

for i in range(0,15):
    maxs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for j in range(minr, maxr):
        #print("SHIT")
        x=0
        scorem=0
        for index1, score in lda_model[bow_corpus[j]]:
            if score>scorem:
                scorem=score
                x=index1
        #print("Topic: {} --- Score: {}".format(max, scorem))
        maxs[x]=maxs[x]+1
    print(str(i)+"->"+str(maxs)+"->"+str(maxs.index(max(maxs))-1));
    maxr=maxr+585
    minr=minr+585

'''
lda_corpus = lda_model[bow_corpus]
scores = list(chain(*[[score for topic_id,score in topic] \
                      for topic in [doc for doc in lda_corpus]]))
threshold = sum(scores)/len(scores)
print("TRESHOLD:"+str(threshold))
cluster=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
cluster[1]= [j for i,j in zip(lda_corpus,bow_corpus) if i[0][1] > threshold]
cluster[2] = [j for i,j in zip(lda_corpus,bow_corpus) if i[1][1] > threshold]
cluster[3] = [j for i,j in zip(lda_corpus,bow_corpus) if i[2][1] > threshold]
cluster[4] = [j for i,j in zip(lda_corpus,bow_corpus) if i[3][1] > threshold]
cluster[5] = [j for i,j in zip(lda_corpus,bow_corpus) if i[4][1] > threshold]
cluster[6] = [j for i,j in zip(lda_corpus,bow_corpus) if i[5][1] > threshold]
cluster[7] = [j for i,j in zip(lda_corpus,bow_corpus) if i[6][1] > threshold]
cluster[8] = [j for i,j in zip(lda_corpus,bow_corpus) if i[7][1] > threshold]
cluster[9] = [j for i,j in zip(lda_corpus,bow_corpus) if i[8][1] > threshold]
cluster[10] = [j for i,j in zip(lda_corpus,bow_corpus) if i[9][1] > threshold]
cluster[11] = [j for i,j in zip(lda_corpus,bow_corpus) if i[10][1] > threshold]
cluster[12] = [j for i,j in zip(lda_corpus,bow_corpus) if i[11][1] > threshold]
cluster[13] = [j for i,j in zip(lda_corpus,bow_corpus) if i[12][1] > threshold]
cluster[14] = [j for i,j in zip(lda_corpus,bow_corpus) if i[13][1] > threshold]
cluster[15] = [j for i,j in zip(lda_corpus,bow_corpus) if i[14][1] > threshold]

for i in range(len(cluster)):
    print(len(cluster[i]))
'''