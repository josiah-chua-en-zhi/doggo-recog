# Overview
This personal MLOps project is to implement an end-to-end pipeline to create a telegram bot that uses deep learning to differentiate dog breeds.
The main objectives are to test the effectiveness of Neptune AI, an ML model monitoring and storage system, and familiarise myself with PyTorch Lightning a Pytorch wrapper package that allows for faster implementation of training deep learning models.

## MLOps Pipeline
![image](https://github.com/josiah-chua/doggo-recog/assets/81459293/8d391a96-ec36-4b51-b1c9-b8c160c4ade7)


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

![image](https://github.com/josiah-chua/doggo-recog/assets/81459293/12ce3fed-7e08-4f2a-a8df-e8be1476350e)


## Preprocessing
Since the current dataset is quite clean, what the preprocessing script current only has a place holder preprocessing funtion. However should user data be stored, there might be a need for futher preprocessing which will be added to the function. This could be a future enhancement

After the preprocessing, the script generate npy array of the photos and stores the numpy array files into the GCS preprocessed bucket and the raw images into the raw image bucket. There should be 4 different GCS buckets, new-raw-images, new-preprocessed-images, raw-images, preprocessed-images. In this first iteration, the buckets with new will not be used as they are used for storing new user data for retraining, and the files should bw stored in the other 2 buckets. 

There are 2 make executors for the preprocessing stage:

will store the preprocessed files into  raw-images, preprocessed-images buckets
```make preprocess_training_files```

```make preprocess_new_files``` : while will store the preprocessed files into new-preprocessed-images, raw-images buckets

## Experimentation
The experimentation section is used because deep learning models take a long time to train, hence to evaluate the effectiveness of ertain models and certain hyperparameters, we can do smaller experimentatiosn with lesser classes and lesser data per class to have a rough gauge of the perormance. While having a better performance in smaller datasets will not gurantee better results in the full dataset, it will be able to give us a sense of how effective it bu looking at the training metrics and prior knowledge of the model archetecture.

This is where pytorch lightning and Neptune.ai come in. Pytorch lightning allows us to package our models nicely and easily attach loggers and callbacks for to monitors our training progress. Training metrics can then be sent ot the Neptune server through the NeptuneLogger and visualised on a dashboard.

For the experimentation stage, one can expeiment with different model archetecture (vgg, efficientnet, resnet) and these have been initalized with ImageNet Weights for transfer learning to reduce the training time needed.

The other hyperparameters to try are:
- image input size (default 224)
- batch size (default 32)
- learning rate (default 1e-3)
- transformation probaility (default = 0.025): this is the chance that the training image will go though a random transformation, more will be explained in the training section.

## Usage
/



# 

# Future Improvements
:green_circle: - Done, :yellow_circle: In Progress, :red_circle: - Yet To Implement

:yellow_circle: Retraining Script

:yellow_circle: Cloud deployment on GCP Kubecluster with docker images

:red_circle: Training on Cloud Compute

:red_circle: Data Collection from users, feedback, and retraining (depending on the data collection policies)
