import requests
import argparse
import neptune
import torch
import telegram
import numpy as np
from PIL import Image
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, Request
from utils_envvar import EnvVar, GeneralUtils
from utils_data import DataLoaderConstructors
from utils_model import Models

"""
Add logging and stuff in the future
"""

API_PORT = 8000

# Initalize Env Vars
env_vars = EnvVar()

class ModelPredictor:
    def __init__(self, model_name):

        self.model_name = model_name
        self.model_repo_id = f"{env_vars.neptune_training_key}-{self.model_name.upper()}"
        self.model_repo = neptune.init_model(
            project = env_vars.neptune_training_project,
            with_id = self.model_repo_id)
    
        model_versions_df = self.model_repo.fetch_model_versions_table().to_pandas()
        production_models = model_versions_df[model_versions_df["sys/stage"] == "production"]
        self.model_id = production_models["sys/id"].sort_values(ascending = True).iloc[0]

        self.model_version = neptune.init_model_version(
            project = env_vars.neptune_training_project,
            with_id = self.model_id
            )

        # load jit model
        self.model_version["model/script"].download(env_vars.deployment_model_dir.as_posix())
        self.model_version["model/labels"].download(env_vars.deployment_model_dir.as_posix())

        self.dataloader_constructors = DataLoaderConstructors(
            filename_col_name = env_vars.local_filenames_col_names,
            label_col_name = env_vars.label_col_names
        )

        self.model = torch.jit.load(env_vars.deployment_model_path.as_posix())
        self.prediction_transformer = self.dataloader_constructors.initalze_prediction_transformer()
        self.label_dict = GeneralUtils.open_json_to_dict(env_vars.deployment_label_path.as_posix())
        print(self.label_dict)

        self.model_repo.stop()
        self.model_version.stop()

    def predict(self):

        # save request data for improving model

        # preprocess/transform data
        image = np.asarray(Image.open(env_vars.tmp_pic_path))
        image = self.prediction_transformer({env_vars.local_filenames_col_names: image,env_vars.label_col_names: None})

        # predict
        pred = self.model(image)
        pred = F.softmax(pred, dim=1)
        pred_top3 = pred.topk(3).indices.tolist()[0]
        pred = pred.tolist()
        pred_pct = [f"{self.label_dict[str(i)]}: {round(100*pred[0][i],2)}%" for i in pred_top3]
        result = f"{pred_pct[0]}\n{pred_pct[1]}\n{pred_pct[2]}"
 
        return result

    @staticmethod
    def save_request_data(image: np.array):
        print("save data step")


class PredictorApi:
    def __init__(self, model_name):
        self.predictor = ModelPredictor(model_name)
        self.app = FastAPI()
        self.bot = telegram.Bot(env_vars.tele_api_key)

        @self.app.get("/")
        async def root():
            return {"message": "No dont touch me there this is my no no square!"}

        @self.app.post("/doggo-recog")
        async def predict_prob1(req: Request):
            print(req)
        
            data = await req.json()

            print(data)

            chat_id, valid_input = await self.parse_message(data)

            if valid_input:
                prediction = self.predictor.predict()
            else:
                prediction = "Please send a picture"

            self.tel_send_message(chat_id, prediction)

            return {"status": "ok"}

    async def parse_message(self, message):
        print("message-->",message)
        chat_id = message['message']['from']['id']
        try:
            file_id = message['message']['photo'][0]["file_id"]
            await self.retrieve_file(file_id)
            valid_input = True
        except KeyError:
            valid_input = False
        print("chat_id-->", chat_id)
        
        return chat_id, valid_input
    
    async def retrieve_file(self, file_id):

        telegram_file = await self.bot.get_file(
            file_id = file_id)

        # Download the file and get the downloaded data as a bytearray
        path = await telegram_file.download_to_drive(env_vars.tmp_pic_path)
        return path


    def tel_send_message(self, chat_id, text):
        url = f"https://api.telegram.org/bot{env_vars.tele_api_key}/sendMessage"
        payload = {
                'chat_id': chat_id,
                'text': text
                }
        r = requests.post(url,json=payload)
        return r

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=Models.model_names)
    args = parser.parse_args()
    # deploy api
    api = PredictorApi(model_name = args.model_name)
    api.run(port=API_PORT)