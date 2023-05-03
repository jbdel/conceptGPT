from collections import Counter
from datetime import datetime
import os
from syslog import LOG_PID
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from image_to_concept.models import BiomedCLIPClassifier
from image_to_concept.datasets import fetch_data
from image_to_concept.utils import get_cutoff_freq


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

EXP_NAME = 'img2concept'

IMAGE_DIR = '/home/cvanuden/git-repos/vilmedic/data/RRG/mimic-cxr/findings'
DATA_DIR = '/home/cvanuden/git-repos/vilmedic/data/RRG/mimic-cxr/concepts'
LOG_DIR = '/home/cvanuden/git-repos/conceptGPT/logs/'
CKPT_DIR = '/home/cvanuden/git-repos/conceptGPT/ckpts/'

CONCEPT_TYPES = ["all_concepts", "concat_concepts"]
THRESHOLDS = [0.001]
# THRESHOLDS = [0.0001, 0.0002, 0.0005, 0.001]

MAX_EPOCHS = 20
PATIENCE = 5
LR=1e-3
NUM_WORKERS = 8


for i, concept_type in enumerate(CONCEPT_TYPES):
    annotations = {}
    for split in ['train', 'validate', 'test']:
        concept_path = os.path.join(DATA_DIR, f'{split}.{concept_type}.tok')
        with open(concept_path) as f:
            split_annotations = f.readlines()
            annotations[split] = {}
            annotations[split][concept_type] = [annot.split(',') for annot in split_annotations]

    for j, threshold in enumerate(THRESHOLDS):
        train_concepts = [string for string_list in annotations['train'][concept_type] for string in string_list]
        c = Counter(train_concepts)
        # get required frequency n to keep concepts that appears at least threshold percent of the time
        n = get_cutoff_freq(c, threshold)
        print(train_concepts[:10], n)
        filtered_concepts = [key for key, value in c.items() if value >= n]
        num_concepts = len(filtered_concepts)
        print(f"exp {i * 4 + j + 1}: concept_type {concept_type}, num_concepts {num_concepts}, exclude threshold {threshold}")
        # thoughts : do some images not have any concepts (and thats okay, but track)
        # fix grammar with gpt ?

        # loggers
        logger_class = getattr(pl_loggers, 'WandbLogger')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger_name = f"{EXP_NAME}_{concept_type}_{threshold}_{timestamp}"
        logger = logger_class(save_dir=LOG_DIR, project='rrg', name=logger_name)

        # callbacks
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(monitor='val_loss', dirpath=os.path.join(CKPT_DIR, logger_name), save_last=True, mode='min', save_top_k=1),
            EarlyStopping(monitor='val_loss', min_delta=0., patience=PATIENCE, mode='min', verbose=False)
        ]

        # Do experiments
        train_dataloader, val_dataloader = fetch_data(filtered_concepts, concept_type, annotations, IMAGE_DIR, num_workers=NUM_WORKERS)
        model = BiomedCLIPClassifier(lr=LR, output_dim=num_concepts)

        trainer = pl.Trainer(callbacks=callbacks, logger=logger, gpus=1, deterministic=True, max_epochs=MAX_EPOCHS)
        trainer.fit(model, train_dataloader, val_dataloader)