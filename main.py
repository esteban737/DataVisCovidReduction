from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import pandas as pd
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)
from sklearn.preprocessing import StandardScaler
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Reduction(BaseModel):
	data: list
	locations: list

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/reduction/pca", response_model=Reduction)
def featureReductionPCA(targets, date):
	df = pd.read_csv('owid-covid-data.csv')
	df = df.loc[df['date'] == date ]
	df1 = df[targets.split(",")]
	df1 = df1.dropna()
	locations = np.array(df1['location'])
	df1.drop(columns=["location"], inplace = True)
	x = np.array(StandardScaler().fit_transform(df1))
	pca = decomposition.PCA(n_components=2)
	x = pca.fit_transform(x)

	return Reduction(data=x.tolist(), locations=locations.tolist())


@app.get("/reduction/lle", response_model=Reduction)
def featureReductionLLE(targets, date):
	df = pd.read_csv('owid-covid-data.csv')
	df = df.loc[df['date'] == date ]
	df1 = df[targets.split(",")]
	df1 = df1.dropna()
	locations = np.array(df1['location'])
	df1.drop(columns=["location"], inplace = True)
	x = np.array(StandardScaler().fit_transform(df1))
	clf = manifold.LocallyLinearEmbedding(n_neighbors=5, n_components=2,
                                      method='standard')
	x = clf.fit_transform(x)

	return Reduction(data=x.tolist(), locations=locations.tolist())