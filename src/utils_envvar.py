"""
Validate format of environemnt varibale and initalize them into
an easily accessible class called Env_Var
"""
import os
import shutil
import logging
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseSettings

from google.cloud import storage

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

load_dotenv()

class GeneralUtils:
    @staticmethod
    def decode_env_var(env_var_string: str):
        """
        Decode a base64 encodede string
        """
        try:
            assert isinstance(env_var_string, str)

            base64_bytes = env_var_string.encode("ascii")
            message_bytes = base64.b64decode(base64_bytes)
            env_var = message_bytes.decode("ascii")
            return env_var
        except base64.binascii.Error:
            print("Decoding ErrorCheck that string has been been base64 encoded")
        except AssertionError:
            print("Type Error: Check that variable to decode is a string")

    @staticmethod
    def save_dict_to_json(filepath, data:dict):
        """
        dafe a dictionary to json file
        """
        with open(filepath, 'w') as file:
            json.dump(data, file)

    @staticmethod
    def create_folder_if_not_exists(folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Folder '{folder_path}' created.")

    @staticmethod
    def delete_directory_contents(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    @staticmethod
    def delete_file(file_path):
        if not os.path.isfile(file_path):
            raise ValueError("source_file_path should be a string")
        try:
            os.unlink(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
    

class EnvVar(BaseSettings):
    """
    Environment variable class
    """
    gcssa_credentials : dict = json.loads(GeneralUtils.decode_env_var(os.environ.get("GCSSA_RAW", None)))
    new_raw_images_bucket_name : str = os.environ.get("NEW_RAW_IMAGES_BUCKET_NAME", None)
    new_processed_images_bucket_name : str = \
        os.environ.get("NEW_PROCESSED_IMAGES_BUCKET_NAME", None)
    processed_images_bucket_name : str = os.environ.get("PROCESSED_IMAGES_BUCKET_NAME", None)
    raw_images_bucket_name : str = os.environ.get("RAW_IMAGES_BUCKET_NAME", None)
    neptune_api_token : str = os.environ.get("NEPTUNE_API_TOKEN", None)
    neptune_exp_project : str = os.environ.get("NEPTUNE_EXP_PROJECT", None)
    neptune_exp_key : str = os.environ.get("NEPTUNE_EXP_PROJECT_KEY", None)
    neptune_training_project : str = os.environ.get("NEPTUNE_TRAINING_PROJECT", None)
    neptune_training_key : str = os.environ.get("NEPTUNE_TRAINING_PROJECT_KEY", None)

    exp_proj_key = {
        "project":neptune_exp_project,
        "key": neptune_exp_key
    }

    training_proj_key = {
        "project":neptune_training_project,
        "key": neptune_training_key
    }


    root_dir = Path(".")
    data_dir = root_dir / "data"
    new_data_dir = data_dir / "new_data"
    temp_data_dir = data_dir / "temp_data"
    model_data_dir = data_dir / "model_data"
    model_data_checkpoints_dir = model_data_dir / "models_checkpoints"
    model_data_models_dir = model_data_dir / "models"
    cred_dir = root_dir / "credentials"

    label_col_names = "label"
    filenames_col_names = "filenames"
    local_filenames_col_names = "local_filenames"

    train_test_val_ratio = 0.25
    random_state = 42

    # create directories becasue they will be excluded in git repo
    GeneralUtils.create_folder_if_not_exists(data_dir)
    GeneralUtils.create_folder_if_not_exists(new_data_dir)
    GeneralUtils.create_folder_if_not_exists(temp_data_dir)
    GeneralUtils.create_folder_if_not_exists(model_data_dir)
    GeneralUtils.create_folder_if_not_exists(model_data_checkpoints_dir)
    GeneralUtils.create_folder_if_not_exists(model_data_models_dir)
    GeneralUtils.create_folder_if_not_exists(cred_dir)

    # Set GOOGLE_APPLICATION_CREDENTIALS with gcssa_credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_dir / "gcssa_credentials.json")
    GeneralUtils.save_dict_to_json(filepath = os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                      data = gcssa_credentials)
    
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_dir / "gcssa_credentials.json")


