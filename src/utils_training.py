"""
Util function for training
"""
import os
from pathlib import Path
from datetime import datetime, timezone
from lightning import LightningModule, Trainer
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.loggers import NeptuneLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torchmetrics
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy
import neptune
from utils_envvar import EnvVar, GeneralUtils
from utils_model import ClassificationModel

class ModelTrainer:
    def __init__(self,
                 env_var: EnvVar,
                 model_name:str,
                 train_dataloader,
                 val_dataloader,
                 test_datloader,
                 proj_key: dict,
                 lr: float,
                 total_labels: int):
        
        self.model_name = model_name
        self.lr = lr
        self.total_labels = total_labels

        self.model_id = f"{model_name}_{datetime.now(timezone.utc).strftime('000000%Y%m%d%H%M%S%f')}"
        self.model_ckpt_path = os.path.join(env_var.model_data_checkpoints_dir,self.model_name,self.model_id)
        self.model_saved_folderpath = os.path.join(env_var.model_data_models_dir, self.model_name)
        self.model_filepath = self.model_id + ".pt"
        self.model_saved_path = os.path.join(self.model_saved_folderpath,self.model_filepath)

        GeneralUtils.create_folder_if_not_exists(self.model_ckpt_path)
        GeneralUtils.create_folder_if_not_exists(self.model_saved_folderpath)

        self.proj = proj_key["project"]
        self.proj_key = proj_key["key"]
        self.model_key = self.model_name.upper()

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_datloader = test_datloader

        self.model = ClassificationModel(
            model_name = self.model_name,
            lr = self.lr,
            total_classes = self.total_labels
            )

    def train_model(self, patience):

        neptune_logger = NeptuneLogger(
            project=self.proj,
            tags=[self.model_name],
            log_model_checkpoints=True
            )

        earlystopping_callback = EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience = patience
            )

        checkpoint_callback = ModelCheckpoint(
            monitor='val_auc',
            dirpath=self.model_ckpt_path,
            filename= self.model_name + '-epoch{epoch:02d}-val_auc{val_auc:.2f}',
            mode = "max",
            auto_insert_metric_name=False,
            save_on_train_epoch_end = True,
            save_top_k = -1,
            save_last = True
            )

        trainer = Trainer(
            default_root_dir=self.model_ckpt_path,
            logger = neptune_logger,
            max_epochs= 24,
            callbacks=[
                earlystopping_callback,
                checkpoint_callback]
            )

        trainer.fit(
            model = self.model,
            train_dataloaders = self.train_dataloader,
            val_dataloaders = self.val_dataloader)

        # production ready model
        script = self.model.to_torchscript()
        torch.jit.save(script, self.model_saved_path)

        best_model = checkpoint_callback.best_model_path

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

        model_version["model/script"].upload(self.model_saved_path)
        model_version["model/checkpoint"].upload(best_model)
        model_version.change_stage('archived')

        model_version.stop()

        if not model_created:
            model_repo.stop()

        neptune_logger.experiment.stop()

# auto change number of workers
# save preprocessor settings

