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
import sys
import warnings
import csv as csv
import heapq

if not sys.warnoptions:
    warnings.simplefilter("ignore")

stopWordsList = []
with open('stopWords.txt') as text_file:
    for line in text_file:
        stopWordsList.append(line.rstrip())

print(stopWordsList)
def read_Test(fname):
    inputfile = csv.reader(open(fname, 'r', encoding="utf-8"))

    num =-1
    for idx, row in enumerate(inputfile):
        if idx %2 ==0:
            num =num+1
            yield gensim.utils.simple_preprocess(row[1])
memeNums = []
stemmer = stemmer('english')
numberOfMemes = 1244
numberOfTestMemes = 585
testPath='meme1sentiment.csv'
trainPath='trainDataNew.csv'
for param1 in range(3,4):
    for param2 in range(15,16):
        for param3 in range(3, 4):
            print("Stop word length:" + str(param1)+". Forget triger size:" + str(param2)+". Epochs:" + str(param3))

            def lemmatize_stemming(text):
                return stemmer.stem(WordNetLemmatizer().lemmatize(text))

            def preprocess(text):
                result = []
                for token in gensim.utils.simple_preprocess(text):
                    if token not in stopWordsList:
                    #if len(token)>2:
                        result.append(lemmatize_stemming(token))
                return result

            processed_docs=[]
            data = pd.read_csv(trainPath, error_bad_lines=False)

            data_text = data[['TEXT']]
            data_text['index'] = data_text.index
            documents = data_text

            #print(documents)
            np.random.seed(2018)
            #nltk.download('wordnet')
            print("Procesing documents...")
            for i in range(len(documents)):
                doc_sample = documents[documents['index'] == i].values[0][0]
                words = []
                #print("Procesing meme" + str(i))
                for word in doc_sample.split(' '):
                    words.append(word)
                temp=preprocess(doc_sample)
                processed_docs.append(temp)

            dictionary = gensim.corpora.Dictionary(processed_docs)
            dictionary.filter_extremes(no_below=param2, no_above=0.5, keep_n=100000)#ovde parametre fitujemo, obavezno u rad
            bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

            tfidf = models.TfidfModel(bow_corpus)
            corpus_tfidf = tfidf[bow_corpus]

            print("LDA training...")
            lda_model = gensim.models.LdaModel(bow_corpus, num_topics=15, id2word=dictionary, passes=param3, minimum_probability=0)
            lda_model.save('ldaTest')
            for idx, topic in lda_model.print_topics(-1):
                print('Topic: {} Words: {}'.format(idx, topic))

            maxr=numberOfMemes-1
            minr=0
            counts=[]
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
                #print(str(i)+"->"+str(maxs)+"->"+str(maxs.index(max(maxs))))
                counts.append(maxs)
                maxr=maxr+numberOfMemes
                minr=minr+numberOfMemes

            maxs = []
            ind = []

            origInd = []

            for i in range(0,15):
                maxs.append(max(counts[i]))
                ind.append(counts[i].index(max(counts[i])))
                origInd.append(np.argsort(counts[i])[::-1][:15])
            for i in range(len(counts)):
                print(counts[i])
            #for i in range(len(ind)):
                #print(ind[i])
            #for i in range(len(origInd)):
                #print(origInd[i])

            #New kul function
            transposedInd = []
            result=[]
            toBeRemoved = []
            whenRemoved = []
            for i in range(0,15):
                transposedInd.append([])
                result.append([])
                toBeRemoved.append([])
                for j in range(0, 15):
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

            for i in range(0,15):

               # print("Ulazak u glavni for : " + str(i))
               # print("UNIEQUE SU : " + str(unique[i]))
                tempUn = unique[i].copy()
                for cluster in tempUn:        #sve unique odmah dodajem
                   # print("TRAZIM UNIQUE : " +str(cluster))
                    for f in range(0, 15):   #nije transponovano pa prolazim ovako kroz sve nizove i gledam
                       # print("F JE : " + str(f))
                       # print( "ORIGINAL IND JE :" + str(origInd[f]))
                      #  print("ORIGINAL IND NA I JE :" + str(origInd[f][i]))
                        if origInd[f][i]==cluster :
                            if f not in paroviDict :
                             #   print("DODAJEM UNIQE : "+ str(f)+" vrednost clustera : " + str(cluster))
                                paroviDict[f] = cluster #meme numb = cluster num
                                for j in range(0, 15): #Izbacujem dodat cluster iz unique
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
                    for h in range(0, 15):   #nije transponovano pa prolazim ovako kroz sve nizove i gledam
                        if origInd[h][i] == conflictingCluster: # trazim indekse
                            if h not in paroviDict :
                                indexes.append(h)
                                conflictingMaxValues.append(counts[h][conflictingCluster]) #same vrednosti
                   # print("U konfliktu su max vrednosti : " + str(conflictingMaxValues))
                    if len(conflictingMaxValues)> 0:
                        index = conflictingMaxValues.index(max(conflictingMaxValues))
                      #  print("DODAJEM KONFLITNAN : " + str(indexes[index]) + " vrednost clustera : " + str(conflictingCluster))
                        paroviDict[indexes[index]] = conflictingCluster

                        for j in range(0, 15):  # Izbacujem dodat cluster iz unique
                            if conflictingCluster in unique[j]:
                                unique[j].remove(conflictingCluster)
                            if conflictingCluster in toBeRemoved[j]:
                                toBeRemoved[j].remove(conflictingCluster)
                            if conflictingCluster in origInd[j]:
                                [-1 if x == conflictingCluster else x for x in origInd[j]]  # menjam uzete vrednosti na -1

                #print("ZA SADA : " + str(paroviDict))


            #print("------------------------------")
            #print(counts)

            #meme je key cluster je value

            success = []

            for key, value in sorted(paroviDict.items()) :
                print("meme" + str(key) + "is in cluster: "+str(counts[key].index(counts[key][value])))
                success.append(counts[key][value]*100/numberOfMemes);
                memeNums.append(str(counts[key].index(counts[key][value])))

            print("Average succes: "+str(sum(success)/len(success))+ " %")
            for idx, rate in enumerate(success) :
                print(str(idx) +". meme success rate is : " + str(rate) + " %")

            processed_docs = []
            test = pd.read_csv(testPath, error_bad_lines=False)
            testBow = []
            test = test[['TEXT']]
            test['index'] = test.index
            testText = test

            print("Processing test memes...")
            for iterator in range(len(testText)):
                doc_sample = testText[testText['index'] == iterator].values[0][0]
                #print(doc_sample)
                #print("Procesing meme" + str(iterator))
                #temp = preprocess(doc_sample)
                #processed_docs.append(temp)
                testBow.append(dictionary.doc2bow(preprocess(doc_sample)))
            #print("HELLO")
            start=0
            end=numberOfTestMemes-1
            testCount = []
            for j in range(0,15):
                testCount.append(0)
                for i in range(start, end):
                    x = 0
                    scorem = 0
                    scoreList = []
                    #print(lda_model[testBow[i]])
                    for index1, score in lda_model[testBow[i]]:

                        #print(score)
                        scoreList.append(score)
                        if score > scorem:
                            scorem = score
                            x = index1

                    maxList = heapq.nlargest(1, range(len(scoreList)), key=scoreList.__getitem__)

                    #+print("Procesing meme in group: " + str(j) + "no: " + str(i)+"Guessed grouo: "+str(x)+". Real group:"+str(memeNums[j]))
                    if maxList[0] == int(memeNums[j]):# or maxList[1] == int(memeNums[j]) or maxList[2] == int(memeNums[j]):# or maxList[3] == int(memeNums[j]):
                        #print("NASO SAM NEKI DA VALJA")
                        testCount[j]=testCount[j]+1
                start=start+numberOfTestMemes
                end=end+numberOfTestMemes
            successTest = []
            print("Average succes: " + str(sum(testCount) * 100 / numberOfTestMemes/15) + " %")
            for i in range(0, 15):
                successTest.append(testCount[i] * 100 / numberOfTestMemes)
                print(str(i) + ". Test meme success rate is : " + str(successTest[i]) + " %")


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