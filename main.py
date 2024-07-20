import pandas as pd
from pymongo import MongoClient
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


client = MongoClient('mongodb+srv://vasudhawaman734:NTmWW8UMpb5980be@cluster0.ctfmgcz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')

databasenames =client.list_database_names()

db =client['music']
collection =db['music']

result =collection.find()

songs =pd.DataFrame(result)
print(songs)
songs['text'] = songs['text'].str.replace(r'\n', '')
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')

lyrics_matrix = tfidf.fit_transform(songs['text'])
cosine_similarities = cosine_similarity(lyrics_matrix)

similarities = {}

for i in range(len(cosine_similarities)):
    
    similar_indices = cosine_similarities[i].argsort()[:-50:-1]
    
    similarities[songs['song'].iloc[i]] = [(cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices][1:]

class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def _print_message(self, song, recom_song):
        rec_items = len(recom_song)

        print(f'The {rec_items} recommended songs for {song} are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_song[i][1]} by {recom_song[i][2]} with {round(recom_song[i][0], 3)} similarity score")
            print("--------------------")

    def recommend(self, recommendation):
        # Get song to find recommendations for
        song = recommendation['song']
        # Get number of songs to recommend
        number_songs = recommendation['number_songs']
        # Get the number of songs most similars from matrix similarities
        recom_song = self.matrix_similar[song][:number_songs]
        # print each item
        #self._print_message(song=song, recom_song=recom_song)
        return recom_song

recommedations = ContentBasedRecommender(similarities)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.post('/recommendContent')
async def recommend(song: str):
    recommendation = {
        "song": song,
        "number_songs": 2
        }
    response=recommedations.recommend(recommendation)
    return JSONResponse(content=response,status_code=200)



# ngrok_tunnel = ngrok.connect(5000)
# print('Public URL:', ngrok_tunnel.public_url)
#nest_asyncio.apply()
#uvicorn.run(app, port=5000)
if __name__ == "__main__":
     uvicorn.run(app, host="0.0.0.0", port=8000)
