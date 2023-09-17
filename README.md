# Overview
This personal MLOps project is to implement an end-to-end pipeline to create a telegram bot that uses deep learning to differentiate dog breeds.
The main objectives are to test the effectiveness of Neptune AI, an ML model monitoring and storage system, and familiarise myself with PyTorch Lightning a Pytorch wrapper package that allows for faster implementation of training deep learning models.

## MLOps Pipeline
![image](https://github.com/josiah-chua/doggo-recog/assets/81459293/73ff80b2-cf8d-4107-8c26-31fea36c80be)


## File structure
```
doggo-recog/
├── credentials/
├── data/
│   ├── deployment_model/
│   ├── model_data/
│   ├── new_data/
│   └── temp_data/
├── src/
│   ├── a_preprocessing.py
│   ├── b_experiment.py
│   ├── c_training.py
│   ├── d_api.py
│   └── utils/
│       ├── utils_data.py
│       ├── utils_envvar.py
│       ├── utils_gcs.py
│       ├── utils_model.py
│       ├── utils_preprocessing.py
│       └── utils_training.py
├── .env
├── venv
├── .gitignore
├── README.md
└── .requirements.txt
```

## Original Dataset
The dataset can be found in the dataset branch. It is from the Standford dogs dataset but the filename have been cleaned up to include the label and a unique id (based on datetime ms) for each photo and has been converted to jpg.
A similar naming convention will also be used in futer implementation of data collection from users.
The dataset has been cleaned beforehand fomatted correctly into its 3 channels (RGB) as a couple of photes were noted to have 4 channels (png photos contain a 4th opacity channel).

![image](https://github.com/josiah-chua/doggo-recog/assets/81459293/8d391a96-ec36-4b51-b1c9-b8c160c4ade7)


## Preprocessing
Since the current dataset is quite clean, what the preprocessing script current only has a place holder preprocessing funtion. However should user data be stored, there might be a need for futher preprocessing which will be added to the function. This could be a future enhancement

After the preprocessing, the script generate npy array of the photos and stores the numpy array files into the GCS preprocessed bucket and the raw images into the raw image bucket. There should be 4 different GCS buckets, new-raw-images, new-preprocessed-images, raw-images, preprocessed-images. In this first iteration, the buckets with new will not be used as they are used for storing new user data for retraining, and the files should bw stored in the other 2 buckets. 

There are 2 make executors for the preprocessing stage:

'''make preprocess_training_files''' : while will store the preprocessed files into  raw-images, preprocessed-images buckets

'''make preprocess_new_files''' : while will store the preprocessed files into  raw-images, preprocessed-images buckets


## Usage
/



# 

# Future Improvements
:green_circle: - Done, :yellow_circle: In Progress, :red_circle: - Yet To Implement

:yellow_circle: Retraining Script

:yellow_circle: Cloud deployment on GCP Kubecluster with docker images

:red_circle: Training on Cloud Compute

:red_circle: Data Collection from users, feedback, and retraining (depending on the data collection policies)
