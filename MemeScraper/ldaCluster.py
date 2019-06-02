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
        print("Topic: {} --- Score: {}".format(max, scorem))
        maxs[x]=maxs[x]+1
    print(str(i)+"->"+str(maxs)+"->"+str(maxs.index(max(maxs))-1));
    maxr=maxr+585
    minr=minr+585

