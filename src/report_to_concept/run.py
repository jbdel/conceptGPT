import os
import json
import tqdm
from sklearn.metrics import f1_score

import src.constants
from src.report_to_concept.utils import annotations_to_concepts, get_annotated_text, get_concepts

# getting all labels (fine-grained - coarse-grained)
all_concepts = []
concat_concepts = []
annotated_texts_with_labels = {}

with open(os.path.join(src.constants.PROJECT_DIR, "src/report_to_concept/test.radgraph.json"), 'r') as f:
    test_annotations = json.load(f)

for annotation in tqdm.tqdm(test_annotations.values()):
    # creating inference radgraph dict from annotation
    new_annotations = {'0': {**annotation.pop("labeler_1"), **{'text': annotation["text"]}}}
    concepts = get_concepts(annotations=new_annotations)

    all_concepts.extend(concepts["all_concepts"])
    concat_concepts.extend(concepts["concat_concepts"])
    annotated_texts_with_labels[get_annotated_text(new_annotations)] = {"all_concepts": concepts["all_concepts"],
                                                                        "concat_concepts": concepts["concat_concepts"]}

all_concepts = list(set(all_concepts))
concat_concepts = list(set(concat_concepts))

all_concept_pred = []
concat_concepts_pred = []
all_concepts_gt = []
concat_concepts_gt = []

# evaluating radgraph
for report, gt_concepts in tqdm.tqdm(annotated_texts_with_labels.items(), total=len(annotated_texts_with_labels)):
    concepts = get_concepts(report=report)
    # pred
    all_concept_pred.append([1 if item in concepts["all_concepts"] else 0 for item in all_concepts])
    concat_concepts_pred.append([1 if item in concepts["concat_concepts"] else 0 for item in concat_concepts])

    # gt
    all_concepts_gt.append([1 if item in gt_concepts["all_concepts"] else 0 for item in all_concepts])
    concat_concepts_gt.append([1 if item in gt_concepts["concat_concepts"] else 0 for item in concat_concepts])


print(f1_score(all_concepts_gt, all_concept_pred, average="micro"))
print(f1_score(concat_concepts_gt, concat_concepts_pred, average="micro"))

print(f1_score(all_concepts_gt, all_concept_pred, average="macro"))
print(f1_score(concat_concepts_gt, concat_concepts_pred, average="macro"))

# 0.851961509992598
# 0.750877192982456
# 0.63318944983074
# 0.5066821072042885