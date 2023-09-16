"""
Util function for training
"""
import os
import torch
import neptune
from datetime import datetime, timezone
from lightning import Trainer
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from utils.utils_envvar import EnvVar, GeneralUtils
from utils.utils_model import ClassificationModel

env_var = EnvVar()

class ModelTrainer:
    def __init__(self,
                 model_name:str,
                 train_dataloader,
                 val_dataloader,
                 test_datloader,
                 proj_key: dict,
                 full_training:bool,
                 lr: float,
                 label_dict: dict,
                 batch_size:int,
                 preprocessing_transforms: dict):
        
        self.model_name = model_name
        self.lr = lr
        self.label_dict = label_dict
        self.total_labels = len(label_dict)
        self.batch_size = batch_size
        self.preprocessing_transform = preprocessing_transforms


        self.model_id = f"{model_name}_{datetime.now(timezone.utc).strftime('000000%Y%m%d%H%M%S%f')}"
        self.model_saved_folderpath = os.path.join(env_var.model_data_models_dir, self.model_name)
        self.model_saved_path = os.path.join(self.model_saved_folderpath,self.model_id + ".pt")
        self.model_label_dict_folderpath = os.path.join(env_var.model_data_label_dict_dir, self.model_name)
        self.model_label_dict_path = os.path.join(self.model_saved_folderpath,self.model_id + "_labels" + ".json")

        GeneralUtils.create_folder_if_not_exists(self.model_saved_folderpath)

        self.proj = proj_key["project"]
        self.proj_key = proj_key["key"]
        self.model_key = self.model_name.upper()
        self.full_training = full_training

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_datloader = test_datloader

        self.model = ClassificationModel(
            model_name = self.model_name,
            model_id = self.model_id,
            lr = self.lr, 
            batch_size = self.batch_size,
            preprocessing_transforms = self.preprocessing_transform,
            total_classes = self.total_labels
            )
        
        self.earlystopping_callback = None
        self.checkpoint_callback = None
        self.neptune_logger = None
        self.trainer = None

        
    def initalize_callbacks(self, patience):

        self.earlystopping_callback = EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience = patience
            )

        self.checkpoint_callback = ModelCheckpoint(
            monitor='val_auc',
            filename= self.model_name + '-epoch{epoch:02d}-val_auc{val_auc:.2f}',
            mode = "max",
            auto_insert_metric_name=False,
            save_top_k = 1,
            save_last = True
            )
        
    def save_artifacts(self, score):
        # production ready model
        script = self.model.to_torchscript()
        torch.jit.save(script, self.model_saved_path)

        # last checkpoint weights for retraining
        best_model = self.checkpoint_callback.best_model_path

        # Check if model repo exists else madke a new one
        try:
            model_repo = neptune.init_model(
                project=self.proj,
                name= self.model_name,
                key= self.model_key)

            model_created = False
        except:
            model_created = True
            print("Model alreaday created")

        model_version = neptune.init_model_version(
            project=self.proj,
            model= self.proj_key + "-" + self.model_key)
        
        # save label dictionary
        GeneralUtils.save_dict_as_json(
            filename = self.model_label_dict_path,
            dict_ = self.label_dict)
        
        model_version["model/script"].upload(self.model_saved_path)
        model_version["model/checkpoint"].upload(best_model)
        model_version["model/labels"].upload(self.model_label_dict_path)
        model_version["model/test_score"] = score[0]
        model_version["model/id"] = {"model-id": self.model_id}
        model_version.change_stage('archived')

        model_version.stop()

        if not model_created:
            model_repo.stop()

        print("Model atrifacts saved to Neptune")


    def train_model(self, max_epochs, patience):

        neptune_logger = NeptuneLogger(
            project=self.proj,
            tags=[self.model_name],
            log_model_checkpoints=True
            )
        
        self.initalize_callbacks(patience)
    
        trainer = Trainer(
            logger = neptune_logger,
            max_epochs= max_epochs,
            log_every_n_steps = len(self.train_dataloader)//2,
            callbacks=[
                self.earlystopping_callback,
                self.checkpoint_callback]
            )

        trainer.fit(
            model = self.model,
            train_dataloaders = self.train_dataloader,
            val_dataloaders = self.val_dataloader)
        
        score = trainer.test(dataloaders=self.test_datloader, ckpt_path='best')

        if self.full_training:
            self.save_artifacts(score)

        neptune_logger.experiment.stop()

# auto change number of workers
# save preprocessor settings

