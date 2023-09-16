import torch
from torch import nn
from lightning import LightningModule
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy
from torchvision.models import efficientnet_b0, resnet18, vgg16



class Models:
    """
    Model Variables
    """
    efficientnet_b0_model = efficientnet_b0(weights="IMAGENET1K_V1")
    resnet18_model = resnet18(weights="IMAGENET1K_V1")
    vgg16_model = vgg16(weights="IMAGENET1K_V1")

    classification_models = {
        "efficientnet": efficientnet_b0_model,
        "resnet": resnet18_model,
        "vgg": vgg16_model
        }
    
    model_names = list(classification_models.keys())
    
    final_layer_access_classfier = ["efficientnet", "vgg"]
    final_layer_access_fc = ["resnet"]

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD

    @staticmethod
    def change_model_final_layer(model_name, model, out_features):

        if model_name in Models.final_layer_access_classfier:
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, out_features)

        elif model_name in Models.final_layer_access_fc:
            in_features = model.fc.in_features
            model.fc= nn.Linear(in_features, out_features)

        else:
            raise ValueError(f"model name {model_name}'s final layer configuration not provided")
        
        return model

class PreprocessingTransforms:

    def __init__(self,
                 input_size = 224,
                 batch_size = 128,
                 lr = 1e-3,
                 transform_probabilities = 0.025):

        self.input_size = input_size
        self.batch_size = batch_size
        self.lr = lr
        self.transform_probabilities = transform_probabilities
        self.rescale_add = self.input_size//7

        self.standard_params = {
            'input_size': self.input_size,
            'rescale_add': self.rescale_add,
            'random_contrast_p': self.transform_probabilities,
            'random_contrast_min_contrast': 0.5,
            'random_contrast_max_contrast': 2.0,
            'random_grey_scale_p': self.transform_probabilities,
            'random_greyscale_ch_num': 3,
            'random_persepctive_p': self.transform_probabilities,
            'random_persepctive_max_padding_ratio': 0.7,
            'random_rorate_p': self.transform_probabilities,
            'random_rorate_rotate_range': (0, 360),
            'random_flip_p': self.transform_probabilities,
            'random_brighten_p': self.transform_probabilities,
            'random_brighten_min_brightness': 0.33,
            'random_brighten_max_brightness': 3,
            'random_gaussian_blur_p': self.transform_probabilities,
            'random_gaussian_blur_kernel_height_range': (7, 12),
            'random_gaussian_blur_kernel_width_range': (7, 12)
        }


class ClassificationModel(LightningModule):
    """
    Lightning module for calssification model
    """
    def __init__(self, 
                 model_name,
                 model_id,
                 lr, 
                 batch_size,
                 preprocessing_transforms,
                 total_classes):
        super().__init__()

        self.model_name = model_name
        self.model_id = model_id
        self.total_classes = total_classes
        self.batch_size = batch_size
        self.preprocessing_transforms = preprocessing_transforms
        self.lr = lr
        self.save_hyperparameters()

        # init a pretrained classification model
        self.model = Models.classification_models[self.model_name]

        self.loss_fn = Models.loss_fn
        self.optimizer = Models.optimizer

        # change the output of the classifier
        self.model = Models.change_model_final_layer(
            model_name = self.model_name,
            model = self.model,
            out_features = self.total_classes)

        # initalize metric for validation
        self.auc_metric = MulticlassAUROC(num_classes=self.total_classes, average="macro", thresholds=None)
        self.acc_metric = MulticlassAccuracy(num_classes=self.total_classes)

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        outputs = self.model(x)

        # Compute the loss and its gradients
        loss = self.loss_fn(outputs, y)
        auc = self.auc_metric(outputs, y)
        acc = self.acc_metric(outputs, y)

        metrics_log = {'train_loss': loss, 'train_auc': auc, "train_acc": acc}
        self.log_dict(
            dictionary = metrics_log,
            prog_bar = True,
            logger = True,
            on_step = False,
            on_epoch = True
        )
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        return optimizer

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y)

        auc = self.auc_metric(outputs, y)
        acc = self.acc_metric(outputs, y)

        metrics_log = {'test_loss': loss, 'test_auc': auc, "test_acc": acc}
        self.log_dict(
            dictionary = metrics_log,
            prog_bar = True,
            logger = True,
            on_step = False,
            on_epoch = True
        )

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y)

        auc = self.auc_metric(outputs, y)
        acc = self.acc_metric(outputs, y)

        metrics_log = {'val_loss': loss, 'val_auc': auc, "val_acc": acc}
        self.log_dict(
            dictionary = metrics_log,
            prog_bar = True,
            logger = True,
            on_step = False,
            on_epoch = True
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)
