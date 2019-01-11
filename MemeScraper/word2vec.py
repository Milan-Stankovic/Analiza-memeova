import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
import os
import collections
import smart_open
import random
import csv


def read_corpus(fname):
    inputfile = csv.reader(open(fname, 'r'))

    for idx, row in enumerate(inputfile):
        if idx %2 ==0:
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(row[1]), [idx])


text = list(read_corpus('index2.csv'))

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

model.build_vocab(text)

#%time
model.train(text, total_examples=model.corpus_count, epochs=model.epochs)

test = model.infer_vector(['Hide', 'the', 'pain','Grba','and', 'git', 'good'])
result = model.docvecs.most_similar([test], topn=len(model.docvecs))

print(result)