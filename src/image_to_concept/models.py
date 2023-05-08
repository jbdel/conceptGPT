"""
MIT License
Copyright (c) 2021 OpenAI
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from collections import OrderedDict
from typing import Tuple, Union

import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
import open_clip

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from .CheXzero import clip


class BaseClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def training_epoch_end(self, training_step_outputs):
        return self.shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self.shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self.shared_epoch_end(test_step_outputs, "test")

    def shared_step(self, batch, split):
        raise NotImplementedError

    def shared_epoch_end(self, step_outputs, split):
        logit = torch.cat([x["logit"] for x in step_outputs])
        label = torch.cat([x["label"] for x in step_outputs])
        prob = torch.sigmoid(logit)

        label = label.detach().cpu().numpy()
        prob = prob.detach().cpu().numpy()

        auroc_list, auprc_list = [], []
        for i in range(label.shape[1]):
            label_cls = label[:, i]
            prob_cls = prob[:, i]

            # check if NaNs or all labels from one class only
            if np.isnan(prob_cls).any() or np.all(np.isclose(label_cls, label_cls[0])):
                auprc_list.append(0)
                auroc_list.append(0)
            else:
                auprc_list.append(average_precision_score(label_cls, prob_cls))
                auroc_list.append(roc_auc_score(label_cls, prob_cls))

        auprc = np.mean(auprc_list)
        auroc = np.mean(auroc_list)

        self.log(f"{split}_auroc", auroc, on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{split}_auprc", auprc, on_epoch=True, logger=True, prog_bar=True)

        # if split == "test":
        #     results_csv = os.path.join(OUTPUT_DIR, "results.csv")
        #     results = {"auroc": auroc, "auprc": auprc}
        #     with open(results_csv, "w") as fp:
        #         json.dump(results, fp)

    def configure_optimizers(self, lr=1e-3, lr_scheduler_factor=0.5, lr_scheduler_patience=5):
        optimizer = optim.Adam([p for p in self.parameters() if p.requires_grad], lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=lr_scheduler_factor, patience=lr_scheduler_patience
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class BiomedCLIPClassifier(BaseClassifier):
    def __init__(self, model_name='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224', freeze_encoder=False, output_dim=2):
        super().__init__()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.model = model.visual
        self.preprocess_train = preprocess_train
        self.preprocess_val = preprocess_val

        # freeze the encoder if applicable
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        self.embed_dim = self.model.trunk.num_features
        self.output_dim = output_dim

        # replace the BioMedCLIP head
        head_layers = OrderedDict()
        head_layers['proj'] = nn.Linear(self.embed_dim, self.output_dim, bias=True)
        self.model.head = nn.Sequential(head_layers)

        self.model.to(device)

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


class CheXzeroClassifier(BaseClassifier):
    def __init__(self, model_path, model_backbone="ViT-B/32", freeze_encoder=False, output_dim=2):
        super().__init__()
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # fetch the pretrained model
        model, preprocess = clip.load(model_backbone, device=device, jit=False)
        model.load_state_dict(torch.load(model_path, map_location=device))

        self.model = model.visual.float()  # IMPT!! CheXzero is in half-precision
        self.preprocess = preprocess

        # freeze the encoder if applicable
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False

        # add a linear head
        self.embed_dim = self.model.output_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(self.embed_dim, self.output_dim, bias=True)
        self.linear = self.linear.to(device)

    def shared_step(self, batch, split):
        # training_step defines the train loop.
        # it is independent of forward
        images = batch["images"].to(self.device)
        labels = batch["labels"].float().to(self.device)

        embeds = self.model(images)
        logits = self.linear(embeds)

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