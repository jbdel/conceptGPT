from collections import Counter
from datetime import datetime
import os
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


# for i, concept_type in enumerate(["all_concepts", "concat_concepts"]):
#   for j, threshold in enumerate([0.0001, 0.0002, 0.0005, 0.001]):
for i, concept_type in enumerate(["all_concepts", "concat_concepts"]):
    concept_path = os.path.join(DATA_DIR, f'train.{concept_type}.tok')
    with open(concept_path) as f:
        annotations = f.readlines()
        annotations = [annot.split(',') for annot in annotations]

    for j, threshold in enumerate([0.001]):
        train_concepts = [string for string_list in annotations for string in string_list]
        c = Counter(train_concepts)
        # get required frequency n to keep concepts that appears at least threshold percent of the time
        n = get_cutoff_freq(c, threshold)
        filtered_concepts = [key for key, value in c.items() if value >= n]
        num_concepts = len(filtered_concepts)
        print(f"exp {i * 4 + j + 1}: concept_type {concept_type}, num_concepts {num_concepts}, exclude threshold {threshold}")
        # thoughts : do some images not have any concepts (and thats okay, but track)
        # fix grammar with gpt ?

        # loggers
        logger_class = getattr(pl_loggers, 'WandbLogger')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        logger_name = f"{EXP_NAME}_{concept_type}_{threshold}_{timestamp}"
        logger = logger_class(save_dir='./logs/', project='rrg', name=logger_name)

        # callbacks
        callbacks = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(monitor='val_loss', dirpath='./data/ckpt', save_last=True, mode='min', save_top_k=1),
            EarlyStopping(monitor='val_loss', min_delta=0., patience=10, mode='min', verbose=False)
        ]

        # Do experiments
        train_dataloader, val_dataloader = fetch_data(filtered_concepts, concept_type, annotations, IMAGE_DIR)
        model = BiomedCLIPClassifier(output_dim=num_concepts)

        trainer = pl.Trainer(callbacks=callbacks, logger=logger, deterministic=True, limit_train_batches=100, max_epochs=1)
        trainer.fit(model, train_dataloader, val_dataloader)