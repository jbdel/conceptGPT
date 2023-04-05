import json
import tqdm
from radgraph import F1RadGraph
from utils import annotations_to_concepts, get_annotated_text

RADGRAPH_SCORER = F1RadGraph(reward_level="all", cuda=-1)
RADGRAPH_MODEL = RADGRAPH_SCORER.radgraph


def get_concepts(report=None, annotations=None):
    assert (report is None) ^ (annotations is None)

    if report is not None:
        annotations = RADGRAPH_MODEL(report)
        assert (len(annotations)) == 1

    return annotations_to_concepts(annotations)


# getting all labels (fine-grained - coarse-grained)
all_concepts = []
concat_concepts = []
annotated_texts = []

test_annotations = json.load(open("chest-xray-testset.json"))
for key in tqdm.tqdm(test_annotations.keys()):
    # creating inference radgraph dict from annotation
    single_annotation = test_annotations[key]
    new_annotations = {'0': {**single_annotation.pop("labeler_1"), **{'text': single_annotation["text"]}}}
    annotated_texts.extend(get_annotated_text(new_annotations))

    concepts = get_concepts(annotations=new_annotations)

    all_concepts.extend(concepts["all_concepts"])
    concat_concepts.extend(concepts["concat_concepts"])

all_concepts = list(set(all_concepts))
concat_concepts = list(set(concat_concepts))


# evaluating radgraph
reports = [
    "increased right lower lobe opacity, concerning for infection. no evidence of pneumothorax.",
    "1 . nodular opacities in the left lower lung with additional small ground-glass opacities bilaterally may represent infection . chest ct recommended for further assessment given infectious symptoms . 2 . abdominal wall varices of indeterminate etiology . 3 . splenomegaly . 4 . coronary artery calcification . acute findings were discussed with dr . ___ by dr . ___ by telephone at 6 : 54 p . m . on ___ ."
]

concepts = get_concepts(report=reports[1])
print(json.dumps(concepts, indent=4))
troll
# test-set annotations
test_annotations = json.load(open("chest-xray-testset.json"))
keys = list(test_annotations.keys())
single_annotation = test_annotations[keys[0]]
new_annotations = {'0': {**single_annotation.pop("labeler_1"), **{'text': single_annotation["text"]}}}
# we add one key about the subtext evaluated by annotators
new_annotations["annotated_text"] = get_annotated_text(new_annotations)

concepts = get_concepts(annotations=new_annotations)
print(json.dumps(concepts, indent=4))
