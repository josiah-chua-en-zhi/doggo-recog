import argparse
import logging
import os
import random
import time

import pandas as pd
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pydantic import BaseModel


API_PORT = 8000

class Data(BaseModel):
    id: str
    rows: list
    columns: list


class ModelPredictor:
    def __init__(self):

        # load jit model
        self.model = None

    def predict(self, data: Data):

        # save request data for improving model

        # preprocess/transform data
        raw_df = pd.DataFrame(data.rows, columns=data.columns)

        # predict
        #prediction = self.model.predict(raw_df)
        prediction = "hehe"
 
        return {"result": prediction}

    @staticmethod
    def save_request_data(image: np.array):
        print("save data step")


class PredictorApi:
    def __init__(self):
        self.predictor = ModelPredictor()
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "No dont touch me there this is my no no square!"}

        @self.app.post("/doggo-recog")
        async def predict_prob1(data: Data, request: Request):
            #self._log_request(request)
            response = self.predictor.predict(data)
            #self._log_response(response)
            return response
        
    @staticmethod
    def _log_request(request: Request):
        pass

    @staticmethod
    def _log_response(response: dict):
        pass

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    # first download deployment mode weights
    
    # deploy api
    api = PredictorApi()
    api.run(port=API_PORT)