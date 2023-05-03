import open_clip

import torch
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl

from collections import OrderedDict
import math
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# define the LightningModule
class BiomedCLIPClassifier(pl.LightningModule):
    def __init__(self, model_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', lr=1e-3, lr_scheduler_factor=0.5, lr_scheduler_patience=5, output_dim=2):
        super().__init__()

        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.model = model.visual
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val

        self.embed_dim = self.model.trunk.num_features
        self.output_dim = output_dim

        # replace the BioMedCLIP head
        head_layers = OrderedDict()
        head_layers['proj'] = nn.Linear(self.embed_dim, self.output_dim, bias=True)
        self.model.head = nn.Sequential(head_layers)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)

        self.lr = lr
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def shared_step(self, batch, split):
        # training_step defines the train loop.
        # it is independent of forward
        images = batch["images"].to(self.device)
        labels = batch["labels"].float().to(self.device)
        logits = self.model(images)

        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
        
        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )

        return_dict = {"loss": loss, "logit": logits.detach(), "label": labels.detach()}
        return return_dict

    def configure_optimizers(self):
        optimizer = optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=self.lr_scheduler_factor, patience=self.lr_scheduler_patience
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}