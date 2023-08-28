
"""
Trianing script
"""
import os
import argparse
import pandas as pd
from utils_envvar import EnvVar, GeneralUtils
from utils_gcs import GCSBucket
from utils_data import DataLoaderConstructors, DataUtils
from utils_model import PreprocessingTransforms, Models
from utils_training import ModelTrainer

# Initalize Env Vars
env_vars = EnvVar()

# Initialize GCS Bucket Names
PROCESSED_IMAGES_BUCKET = GCSBucket(bucket_name = env_vars.new_processed_images_bucket_name)

class Training:

    @staticmethod
    def get_dataset_n_label_dict(n_labels:int, n_samples:int, all_data: bool = True):

        dataset_df = PROCESSED_IMAGES_BUCKET.read_metadata_csv()

        if all_data:
            sample_dataset, label_dict = DataUtils.get_label_dict_n_relabel_to_int(
                dataset_df = dataset_df,
                label_col_name = env_vars.label_col_names,
                filenames_col_name = env_vars.filenames_col_names)
            
        else:
            sample_dataset, label_dict = DataUtils.get_random_filenames_by_label(
                dataset_df = dataset_df,
                label_col_name = env_vars.label_col_names,
                filenames_col_name = env_vars.filenames_col_names,
                n_labels = n_labels,
                n_samples = n_samples)
            
        sample_dataset[env_vars.local_filenames_col_names] = sample_dataset[env_vars.filenames_col_names].apply(\
        lambda filename: os.path.join(env_vars.temp_data_dir, filename))
    
        sample_dataset.to_csv(os.path.join(env_vars.temp_data_dir, "meta_data.csv"),index=False)

        return sample_dataset, label_dict
    
    @staticmethod
    def download_data(dataset_df: pd.DataFrame):

        count = 0
        # Download into temp file
        for filename, local_filename in zip(dataset_df[env_vars.filenames_col_names], \
                                            dataset_df[env_vars.local_filenames_col_names]):
            
            PROCESSED_IMAGES_BUCKET.download_file(filename, local_filename)
            count+=1

            if count%100 == 0:
                print(f"{count}/{len(dataset_df)} files downloaded")

        print(f"{count}/{len(dataset_df)} files downloaded")

    
    @staticmethod
    def train_model(
        model_name:str,
        patience:int,
        n_labels:int,
        n_samples:int,
        full_training: bool = False):

        if full_training:
            proj_key = env_vars.training_proj_key
        else:
            proj_key = env_vars.exp_proj_key

        # get dataset df and labe dictionary
        dataset_df, label_dict = Training.get_dataset_n_label_dict(
            n_labels = n_labels, n_samples = n_samples, all_data = full_training)
        
        # download training data
        Training.download_data(dataset_df)

        train_dataset_df, val_dataset_df, test_dataset_df = DataUtils.train_val_test_split(
            dataset_df = dataset_df,
            split_ratio = env_vars.train_test_val_ratio,
            label_col_name = env_vars.label_col_names,
            random_state = env_vars.random_state
            )
        
        dataloader_constructors = DataLoaderConstructors(
            train_dataset = train_dataset_df,
            val_dataset = val_dataset_df,
            test_dataset =  test_dataset_df,
            filename_col_name = env_vars.local_filenames_col_names,
            label_col_name = env_vars.label_col_names
        )
        train_dataloader, val_dataloader, test_dataloader = dataloader_constructors.initalize_dataloaders(
            **PreprocessingTransforms.standard_params
        )

        model_trainer = ModelTrainer(
            env_var = env_vars,
            model_name = model_name,
            train_dataloader = train_dataloader,
            val_dataloader = val_dataloader,
            test_datloader = test_dataloader,
            proj_key = proj_key,
            lr = PreprocessingTransforms.standard_lr,
            total_labels = len(label_dict)
            )
        
        model_trainer.train_model(patience = patience)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=Models.model_names)
    parser.add_argument("--patience", type = int, default= 5)
    parser.add_argument("--n-labels", type = int, default= 10)
    parser.add_argument("--n-samples", type = int, default= 100)
    parser.add_argument("--delete-temp", action="store_true")
    args = parser.parse_args()

    Training.train_model(
        model_name = args.model_name,
        patience = args.patience,
        n_labels = args.n_labels,
        n_samples = args.n_samples,
        full_training = False
    )

    if args.delete_temp:
        GeneralUtils.delete_directory_contents(env_vars.temp_data_dir)

