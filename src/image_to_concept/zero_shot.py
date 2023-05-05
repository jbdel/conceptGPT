from collections import Counter
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

import CheXzero.eval as eval
import CheXzero.zero_shot as zero_shot
from utils import get_cutoff_freq

DATA_DIR = '/home/cvanuden/git-repos/vilmedic/data/RRG/mimic-cxr/concepts'
CONCEPT_TYPES = ["all_concepts", "concat_concepts"]
THRESHOLDS = [0.001]

## Define Zero Shot Labels and Templates

# ----- DIRECTORIES ------ #
cxr_filepath: str = 'CheXzero/data/cxr.h5' # filepath of chest x-ray images (.h5)
cxr_true_labels_path: Optional[str] = 'CheXzero/data/groundtruth.csv' # (optional for evaluation) if labels are provided, provide path
model_dir: str = '/home/cvanuden/git-repos/conceptGPT/ckpts/chexzero' # where pretrained models are saved (.pt) 
predictions_dir: Path = Path('CheXzero/predictions') # where to save predictions
cache_dir: str = predictions_dir / "cached" # where to cache ensembled predictions

context_length: int = 77

# ------- LABELS ------  #
# Define labels to query each image | will return a prediction for each label
cxr_labels: List[str] = []

# ---- TEMPLATES ----- # 
# Define set of templates | see Figure 1 for more details                        
cxr_pair_template: Tuple[str] = ("{}", "no {}")

# ----- MODEL PATHS ------ #
# If using ensemble, collect all model paths
model_paths = []
for subdir, dirs, files in os.walk(model_dir):
    for file in files:
        full_dir = os.path.join(subdir, file)
        model_paths.append(full_dir)


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

        # ------- LABELS ------  #
        # Define labels to query each image | will return a prediction for each label
        cxr_labels: List[str] = filtered_concepts

        # process ground truth for input images
        data = []
        for i,row in enumerate(annotations['train'][concept_type]):
            one_hot = np.zeros((len(cxr_labels),))
            for j,label in enumerate(cxr_labels):
                if label in row:
                    one_hot[j] = 1
            data.append(one_hot)

        df = pd.DataFrame(data, columns=cxr_labels)
        df.to_csv(cxr_true_labels_path, index=False)

        # ------- EXPERIMENTS! ------  #
        # computes predictions for a set of images stored as a np array of probabilities for each pathology
        predictions, y_pred_avg = zero_shot.ensemble_models(
            model_paths=model_paths, 
            cxr_filepath=cxr_filepath, 
            cxr_labels=cxr_labels, 
            cxr_pair_template=cxr_pair_template, 
            cache_dir=cache_dir,
        )

        # loads in ground truth labels into memory
        test_pred = y_pred_avg
        test_true = zero_shot.make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

        # evaluate model, no bootstrap
        cxr_results: pd.DataFrame = eval.evaluate(test_pred, test_true, cxr_labels) # eval on full test datset

        # boostrap evaluations for 95% confidence intervals
        bootstrap_results: Tuple[pd.DataFrame, pd.DataFrame] = eval.bootstrap(test_pred, test_true, cxr_labels) # (df of results for each bootstrap, df of CI)

        # print results with confidence intervals
        print(bootstrap_results[1])