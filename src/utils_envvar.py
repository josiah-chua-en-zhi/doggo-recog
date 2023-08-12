"""
Validate format of environemnt varibale and initalize them into
an easily accessible class called Env_Var
"""
import os
import logging
import base64
import json
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseSettings

from google.cloud import storage

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

load_dotenv()

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

def save_dict_to_json(filepath, data:dict):
    """
    dafe a dictionary to json file
    """
    with open(filepath, 'w') as file:
        json.dump(data, file)
    

class EnvVar(BaseSettings):
    """
    Environment variable class
    """
    gcssa_credentials : dict = json.loads(decode_env_var(os.environ.get("GCSSA_RAW", None)))
    new_raw_images_bucket_name : str = os.environ.get("NEW_RAW_IMAGES_BUCKET_NAME", None)
    new_processed_images_bucket_name : str = \
        os.environ.get("NEW_PROCESSED_IMAGES_BUCKET_NAME", None)
    processed_images_bucket_name : str = os.environ.get("PROCESSED_IMAGES_BUCKET_NAME", None)
    raw_images_bucket_name : str = os.environ.get("RAW_IMAGES_BUCKET_NAME", None)
    experiment_classes : int = os.environ.get("EXPERIMENT_CLASSES", None)
    experiment_samples_each_class : int = os.environ.get("EXPERIMENT_SAMPLES_EACH_CLASS", None)
    neptune_api_token : str = os.environ.get("NEPTUNE_API_TOKEN", None)
    neptune_exp_project : str = os.environ.get("NEPTUNE_EXP_PROJECT", None)

    root_dir = Path(".")
    data_dir = root_dir / "data"
    new_data_dir = data_dir / "new_data"
    temp_data = data_dir / "temp_data"

    cred_dir = root_dir / "credentials"

    # Set GOOGLE_APPLICATION_CREDENTIALS with gcssa_credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(cred_dir / "gcssa_credentials.json")
    save_dict_to_json(filepath = os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
                      data = gcssa_credentials)


