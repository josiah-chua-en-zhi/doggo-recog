""""
Preprocessign script
"""
import os
import sys
import argparse
from utils.utils_envvar import EnvVar, GeneralUtils
from utils.utils_gcs import GCSBucket
from utils.utils_preprocessing import PreprocessingUtils, ImageProcessor


# Initalize Env Vars
env_vars = EnvVar()

NEW_RAW_IMAGES_BUCKET = GCSBucket(bucket_name = env_vars.new_raw_images_bucket_name)
NEW_PROCESSED_IMAGES_BUCKET = GCSBucket(bucket_name = env_vars.new_processed_images_bucket_name)

class Preprocessing:
    image_preprocessor = ImageProcessor()

    @staticmethod
    def preprocess_new_data(max_data: int, full_dataset : bool = False, delete_files : bool = False):

        # intialise iterator to go though all new images
        new_data_iter = PreprocessingUtils.get_new_img_arrays(env_vars.new_data_dir)
        #check system size
        sys.getsizeof(new_data_iter)

        count = 0

        for image_filename, image_array in new_data_iter:

            local_file_path = env_vars.new_data_dir / image_filename

            # get the dog type
            image_filename_list = image_filename.split("_")
            # some names have _ inside their name
            if len(image_filename_list) <= 2:
                type_subfolder = image_filename_list[0]
            else:
                type_subfolder = '_'.join(image_filename_list[:-1])

            # get new filenames
            processed_destination_filename = PreprocessingUtils.change_filepath_file_type(
                filename = image_filename,
                new_file_type = "npy")
            processed_destination_filepath = os.path.join(type_subfolder, 
                                                          processed_destination_filename)
            raw_destination_filepath = os.path.join(type_subfolder, image_filename)

            # preprocess images
            processed_image = Preprocessing.image_preprocessor(image_array)

            image_bytes = PreprocessingUtils.convert_image_from_numpy_to_bytes(processed_image)


            NEW_PROCESSED_IMAGES_BUCKET.save_data(
                file_path = processed_destination_filepath,
                content = image_bytes)

            NEW_RAW_IMAGES_BUCKET.upload_file(
                source_file_path = local_file_path,
                destination_blob_name = raw_destination_filepath,
            )
            print(f"{processed_destination_filepath} saved")

            if delete_files:
                GeneralUtils.delete_file(local_file_path)

            count += 1
            if count == max_data and not full_dataset:
                break
            
        NEW_PROCESSED_IMAGES_BUCKET.update_metadata_csv()
        NEW_RAW_IMAGES_BUCKET.update_metadata_csv()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-data", type = int, default= 5000)
    parser.add_argument("--full-dataset", action="store_true")
    parser.add_argument("--delete-local", action="store_true")
    args = parser.parse_args()

    Preprocessing.preprocess_new_data(
        max_data = args.max_data,
        full_dataset = args.full_dataset,
        delete_files= args.delete_local)

