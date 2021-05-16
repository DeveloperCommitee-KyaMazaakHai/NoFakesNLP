import json
import time
import random
import pandas as pd
import numpy as np
import requests as rq
from essential_generators import DocumentGenerator


# YOUR INPUT SENTENCE
testSentence = "Calculus seems is a nice thing"


getEmbeddingsURL = "http://127.0.0.1:5000/computeEmbedding"
getFakeFactorURL = "http://127.0.0.1:5000/computeDistance"


def cosine(a,b):
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

def getDatabaseEntry(sentence):
    params={"message":sentence}
    embedding = rq.post(url=getEmbeddingsURL,json=params).json()
    embedding = list(embedding)
    hit = random.randint(1,100_000)
    return embedding, hit

def getDatabaseRandom(testLimit):
    sentenceGenerator = DocumentGenerator()
    databaseSentences= []
    databaseHits = []
    databaseEmbeddings = []
    for i in range(testLimit):
        sentence = sentenceGenerator.sentence()
        embedding,hit = getDatabaseEntry(sentence)
        databaseHits.append(hit)
        databaseEmbeddings.append(embedding)
        databaseSentences.append(sentence)
    return databaseSentences,databaseEmbeddings,databaseHits

def getDatabaseFixed():
    df = pd.read_csv('sentences.csv')

    databaseSentences = []
    databaseHits = []
    databaseEmbeddings = []

    for i in range(len(df)):
        sentence = df['Sentence'][i]
        hit = int(df['Hit'][i])
        embedding,_ = getDatabaseEntry(sentence)
        databaseHits.append(hit)
        databaseEmbeddings.append(embedding)
        databaseSentences.append(sentence)
    return databaseSentences,databaseEmbeddings,databaseHits


databaseSentences,databaseEmbeddings,databaseHits = getDatabaseFixed()
currentEmbedding = rq.post(url=getEmbeddingsURL,json={"message":testSentence}).json()
dbParams ={"currentEmbedding":currentEmbedding,"databaseEmbeddings":databaseEmbeddings,"databaseHits":databaseHits,"databaseSentences":databaseSentences}

ans = rq.post(url=getFakeFactorURL,json=dbParams)