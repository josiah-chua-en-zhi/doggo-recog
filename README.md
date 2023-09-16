# Overview
This personal MLOps project is to implement an end-to-end pipeline to create a telegram bot that uses deep learning to differentiate dog breeds.
The main objectives are to test the effectiveness of Neptune AI, an ML model monitoring and storage system, and familiarise myself with PyTorch Lightning a Pytorch wrapper package that allows for faster implementation of training deep learning models.

## MLOps Pipeline
![image](https://github.com/josiah-chua/doggo-recog/assets/81459293/73ff80b2-cf8d-4107-8c26-31fea36c80be)


## File structure
```
doggo-recog/
├── src/
│   ├── a_preprocessing.py
│   ├── b_experiment.py
│   ├── c_training.py
│   └── d_api.py
│       ├── utils/
│       ├── utils_data.py
│       ├── utils_envvar.py
│       ├── utils_gcs.py
│       ├── utils_model.py
│       ├── utils_preprocessing.py
│       └── utils_training.py
├── .gitignore
├── README.md
├── .requirements.txt
```

## Original Dataset
The dataset can be found in the dataset branch. It is from the standford dogs dataset but the filename have been cleaned up to include the label and a unique id (based on datetime ms) for each photo.
A similar naming convention will also be used in futer implementation of data collection from users.

![image](https://github.com/josiah-chua/doggo-recog/assets/81459293/5bdcb5b7-6d2f-479c-8c75-9e9ab8e570d1)


## Preprocessing
Since the current dataset is quite clean, what the preprocessing script current does is just to check if the pictures can be opened and be fomatted correctly into its 3 channels (RGB) as a couple of photes were noted to have 4 channels (png photos contain a 4th opacity channel).The 4th channel would require an additional preprocessing function, however as this dataset had very few, I opted to just drop the picture, and convert it all into jpg to prevent such issues. This could be a future enhancement


## Usage
/



# 

# Future Improvements
:green_circle: - Done, :yellow_circle: In Progress, :red_circle: - Yet To Implement

:yellow_circle: Retraining Script

:yellow_circle: Cloud deployment on GCP Kubecluster with docker images

:red_circle: Training on Cloud Compute

:red_circle: Data Collection from users, feedback, and retraining (depending on the data collection policies)
