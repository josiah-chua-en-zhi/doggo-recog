"""
Experiment script
"""
import os
import argparse
import pandas as pd
from utils.utils_envvar import EnvVar, GeneralUtils
from utils.utils_gcs import GCSBucket
from utils.utils_data import DataLoaderConstructors, DataUtils
from utils.utils_model import PreprocessingTransforms, Models
from utils.utils_training import ModelTrainer

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
        max_epochs:int,
        patience:int,
        n_labels:int,
        n_samples:int,
        input_size: int = 224,
        batch_size: int = 128,
        lr: float = 0.001,
        transform_probabilities: float = 0.025,
        full_training: bool = False):

        if full_training:
            proj_key = env_vars.training_proj_key
        else:
            proj_key = env_vars.exp_proj_key
            
        preprocessing_transforms = PreprocessingTransforms(
            input_size = input_size,
            batch_size = batch_size,
            lr = lr,
            transform_probabilities = transform_probabilities)

        # get dataset df and label dictionary
        dataset_df, label_dict = Training.get_dataset_n_label_dict(
            n_labels = n_labels, n_samples = n_samples, all_data = full_training)
        
        # download training data
        Training.download_data(dataset_df)

        train_dataset_df, val_dataset_df, test_dataset_df = DataUtils.train_val_test_split(
            dataset_df = dataset_df,
            split_ratio = env_vars.train_test_val_ratio,
            label_col_name = env_vars.label_col_names
            )
        
        dataloader_constructors = DataLoaderConstructors(
            train_dataset = train_dataset_df,
            val_dataset = val_dataset_df,
            test_dataset =  test_dataset_df,
            filename_col_name = env_vars.local_filenames_col_names,
            label_col_name = env_vars.label_col_names
        )

        dataloader_constructors.initalize_data_transformers(
            **preprocessing_transforms.standard_params
        )

        train_dataloader, val_dataloader, test_dataloader = dataloader_constructors.initalize_dataloaders(
            batch_size = preprocessing_transforms.batch_size
        )

        model_trainer = ModelTrainer(
            model_name = model_name,
            train_dataloader = train_dataloader,
            val_dataloader = val_dataloader,
            test_datloader = test_dataloader,
            proj_key = proj_key,
            full_training = True, #rmb to change back later 
            lr = preprocessing_transforms.lr,
            label_dict = label_dict,
            batch_size = preprocessing_transforms.batch_size,
            preprocessing_transforms = preprocessing_transforms.standard_params,
            )
        
        model_trainer.train_model(max_epochs = max_epochs, patience = patience)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=Models.model_names)
    parser.add_argument("--patience", type = int, default= 5)
    parser.add_argument("--max-epochs", type = int, default= 64)
    parser.add_argument("--n-labels", type = int, default= 10)
    parser.add_argument("--n-samples", type = int, default= 100)
    parser.add_argument("--input-size", type = int, default= 224)
    parser.add_argument("--batch-size", type = int, default= 32)
    parser.add_argument("--lr", type = float, default= 0.001)
    parser.add_argument("--trf-prb", type = float, default= 0.025)
    parser.add_argument("--delete-temp", action="store_true")
    args = parser.parse_args()

    Training.train_model(
        model_name = args.model_name,
        max_epochs = args.max_epochs,
        patience = args.patience,
        n_labels = args.n_labels,
        n_samples = args.n_samples,
        input_size = args.input_size,
        batch_size = args.batch_size,
        lr = args.lr,
        transform_probabilities = args.trf_prb,
        full_training = False
    )

    if args.delete_temp:
        GeneralUtils.delete_directory_contents(env_vars.temp_data_dir)

