from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
df = pd.read_csv('owid-covid-data.csv')
class CleanData(BaseModel):
	ndata: list
	data: list
	locations: list
	
class ReqBody(BaseModel):
	x : list
	perplexity : Optional[float] = 30
	learningrate : Optional[float] = 200
	n_neighbors : Optional[int] = 5

# class AllReduction(BaseModel):
# 	cleanData: list
# 	locations :list
# 	pca: list
# 	lle: list
# 	tsne: list


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/reduction/clean", response_model=CleanData)
async def clean(targets,date):
	df = pd.read_csv('owid-covid-data.csv')
	df1 = df.loc[df['date'] == date ]
	df1 = df1[targets.split(",")]
	df1 = df1.dropna()
	
	worldindex = df1[df1['location']=="World"].index.values.astype(int)
	
	if (len(worldindex) == 1): 
		df1 = df1.drop(index=worldindex[0])

	locations = np.array(df1['location'])
	df1.drop(columns=["location"], inplace = True)
	x = np.array(df1)
	x1 = np.array(StandardScaler().fit_transform(df1))
	df = df[0:0]
	
	return CleanData(ndata =x1.tolist() ,data = x.tolist(), locations = locations.tolist())

@app.post("/reduction/tsne")
def featureReductionTSNE(req : ReqBody):
	x = np.array(req.x)
	tsne = manifold.TSNE(n_components=2, init='pca', random_state=0, perplexity = req.perplexity, learning_rate = req.learningrate)
	x = tsne.fit_transform(x)

	return x.tolist()

@app.post("/reduction/pca")
def featureReductionPCA(req : ReqBody):
	x = np.array(req.x)
	pca = decomposition.PCA(n_components=2)
	x = pca.fit_transform(x)

	return x.tolist()


@app.post("/reduction/lle")
def featureReductionLLE(req : ReqBody):
	x = np.array(req.x)
	clf = manifold.LocallyLinearEmbedding(n_neighbors=req.n_neighbors, n_components=2,
                                      method='standard')
	x = clf.fit_transform(x)

	return x.tolist()
