import json
import heapq
import numpy as np

from numba import njit
from numpy.linalg import norm
from flask import Flask,request
from sentence_transformers import SentenceTransformer


app = Flask(__name__)
model = SentenceTransformer('bert-base-nli-mean-tokens')

@njit
def normalize(array):
  array = (array-np.amin(array))/(np.amax(array)-np.amin(array))
  return array

def computeDistances(currentEmbedding,databaseEmbeddings,databaseHits,databaseSentences,nNearest):
  
  frame = []
  
  # Find distance of all embeddings and store top 10 similar ones in 'frame'
  for i in range(len(databaseEmbeddings)):
    similarity = np.dot(currentEmbedding, databaseEmbeddings[i])/(norm(currentEmbedding)*norm(databaseEmbeddings[i]))
    if(len(frame)==nNearest):
      heapq.heappushpop(frame,(similarity,databaseHits[i],i))
    elif(len(frame)<nNearest):
      heapq.heappush(frame,(similarity,databaseHits[i],i))
  
  
  # Split 'frame' into similarity, hits, indexes arrays
  similarities = []
  hits = []
  indexes = []

  while(len(frame)>0):
    item = heapq.heappop(frame)
    similarities.append(item[0])
    hits.append(item[1])
    indexes.append(item[2])

  # Convert similarity to numpy - they are already on scale of 0-1
  similarities = np.asarray(similarities)
  
  # normalised Hits first takes distance to power^6  => very close ones are given more weightage and very far ones are given much lesser weightage
  # this distance powered^6 is multiplied with hits to give weightedHits for each entry
  # This weightedHits is normalized to give normalised Hits
  hits = np.asarray(hits)
  normalisedHits = np.multiply(normalize(np.power(similarities,6)),hits)
  normalisedHits = normalize(normalisedHits)

  
  weightedHits = np.multiply(similarities,normalisedHits)
  
  print("index\tS\tH\tNH\tmult\tsentence")
  for i in range(len(similarities)):
    print(str(indexes[i])+"\t"+f"{similarities[i]:.4f}"+"\t"+str(hits[i])+"\t"+f"{normalisedHits[i]:.4f}"+"\t"+f"{weightedHits[i]:.4f}"+"\t"+databaseSentences[indexes[i]]) 
  
  ans = np.sum(weightedHits)/np.sum(normalisedHits)
  print("Fake probability: "+str(ans*100))



@app.route("/computeEmbedding",methods=["POST"])
def computeEmbedding():
    requestJson = request.get_json(force=True)
    message = requestJson["message"]
    embedding = model.encode([message])[0].tolist()
    embedding = json.dumps(embedding)
    return embedding

@app.route("/computeDistance",methods=["POST"])
def computeFakeness():
  requestJson = request.get_json(force=True)
  currentEmbedding = np.array(requestJson["currentEmbedding"])
  databaseEmbeddings = np.asarray(requestJson["databaseEmbeddings"])
  databaseHits = np.array(requestJson["databaseHits"])
  databaseSentences = requestJson["databaseSentences"]
  computeDistances(currentEmbedding,databaseEmbeddings,databaseHits,databaseSentences,5)
  return "bruh"
    
app.run()