# Overview
This personal MLOps project is to implement an end-to-end pipeline to create a telegram bot that uses deep learning to differentiate dog breeds.
The main objectives are to test the effectiveness of Neptune AI, an ML model monitoring and storage system, and familiarise myself with PyTorch Lightning a Pytorch wrapper package that allows for faster implementation of training deep learning models.

## MLOps Pipeline
![DoggoRecogo drawio](https://github.com/josiah-chua/doggo-recog/assets/81459293/3788d92f-7261-444a-b4ae-9a1f38bb11f7)

## File structure
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


## Original Dataset
The dataset can be found in the dataset branch 
## Usage
/



# 

# Future Improvements
:green_circle: - Done, :yellow_circle: In Progress, :red_circle: - Yet To Implement

:yellow_circle: Cloud deployment on GCP Kubecluster with docker images

:red_circle: Training on Cloud Compute

:red_circle: Data Collection from users, feedback, and retraining (depending on the data collection policies)
