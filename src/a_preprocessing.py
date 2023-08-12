""""
Preprocessign script
"""
import os
import sys
import inspect
from io import BytesIO
import numpy as np
import pandas as pd
from utils_envvar import EnvVar
from utils_gcs import GCSBucket
from utils_preprocessing import PreprocessingUtils, ImageProcessor


# Initalize Env Vars
env_vars = EnvVar()

NEW_RAW_IMAGES_BUCKET = GCSBucket(bucket_name = env_vars.new_raw_images_bucket_name)
NEW_PROCESSED_IMAGES_BUCKET = GCSBucket(bucket_name = env_vars.processed_images_bucket_name)

class Preprocessing:
    image_preprocessor = ImageProcessor
    @staticmethod
    def preprocess_new_data():

        # intialise iterator to go though all new images
        new_data_iter = PreprocessingUtils.get_new_img_arrays(env_vars)
        #check system size
        sys.getsizeof(new_data_iter)

        for image_filepath, image_array in new_data_iter:

            # get the dog type
            # some names have _ inside their name
            image_filepath_list = image_filepath.split("_")
            if len(image_filepath_list) <= 2:
                type_subfolder = image_filepath_list[0]
            else:
                type_subfolder = '_'.join(image_filepath_list[:-1])

            processed_destination_filename = PreprocessingUtils.change_filepath_file_type(
                filename = image_filepath,
                new_file_type = ".npy")
            processed_destination_filepath = os.path.join(type_subfolder, 
                                                          processed_destination_filename)

            raw_destination_filepath = os.path.join(type_subfolder, image_filepath)

            processed_image = Preprocessing.image_preprocessor(image_array)
            image_bytes = PreprocessingUtils.convert_image_from_numpy_to_bytes(processed_image)

            NEW_PROCESSED_IMAGES_BUCKET.save_data(
                file_path = processed_destination_filepath, 
                content = image_bytes)

            NEW_RAW_IMAGES_BUCKET.upload_file(
                source_file_path = image_filepath, 
                destination_blob_name = raw_destination_filepath,
            )
            print(f"{processed_destination_filepath} saved")

        NEW_PROCESSED_IMAGES_BUCKET.update_metadata_csv()
        NEW_RAW_IMAGES_BUCKET.update_metadata_csv()

if __name__ == '__main__':

    Preprocessing.preprocess_new_data()
