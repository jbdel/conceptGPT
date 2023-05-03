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
    def __init__(self, model_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', lr=1e-3, output_dim=2):
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
        self.sigm = nn.Sigmoid()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)

        self.lr = lr

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam([p for p in self.parameters() in p.requires_grad()], lr=self.lr)
        return optimizer