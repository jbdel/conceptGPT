<h2>report_to_concept</h2>

There are two tools:

#### 1. `get_concept()`

```python
import json
from utils import annotations_to_concepts

def get_concepts(report=None, annotations=None):
    assert (report is None) ^ (annotations is None)

    if report is not None:
        annotations = RADGRAPH_MODEL(report)
        assert (len(annotations)) == 1

    return annotations_to_concepts(annotations)
    
report = "1 . nodular opacities in the left lower lung with additional small ground-glass opacities bilaterally may represent infection . chest ct recommended for further assessment given infectious symptoms . 2 . abdominal wall varices of indeterminate etiology . 3 . splenomegaly . 4 . coronary artery calcification . acute findings were discussed with dr . ___ by dr . ___ by telephone at 6 : 54 p . m . on ___ ."

concepts = get_concepts(report=report)
print(json.dumps(concepts, indent=4))
```

outputs

```json
{
  "processed_annotations": [
    ...
  ],
  "radgraph_annotations": {
    "0": {
      "text": "1 . nodular opacities in the left lower lung with additional small ground-glass opacities bilaterally may represent infection . chest ct recommended for further assessment given infectious symptoms . 2 . abdominal wall varices of indeterminate etiology . 3 . splenomegaly . 4 . coronary artery calcification . acute findings were discussed with dr . ___ by dr . ___ by telephone at 6 : 54 p . m . on ___ .",
      "entities": {
        ...
      }
    }
  },
  "radgraph_text": "1 . nodular opacities in the left lower lung with additional small ground-glass opacities bilaterally may represent infection . chest ct recommended for further assessment given infectious symptoms . 2 . abdominal wall varices of indeterminate etiology . 3 . splenomegaly . 4 . coronary artery calcification . acute findings were discussed with dr . ___ by dr . ___ by telephone at 6 : 54 p . m . on ___ .",
  "all_concepts": [
    "calcification",
    "left lower lung",
    "abdominal wall",
    "splenomegaly",
    "possible infection",
    "varices indeterminate etiology",
    "coronary artery",
    "nodular opacities",
    "small opacities",
    "opacities suggestive of infection"
  ],
  "concat_concepts": [
    "nodular opacities located at left lower lung suggestive of infection",
    "small opacities suggestive of infection",
    "varices indeterminate etiology located at abdominal wall",
    "splenomegaly",
    "calcification located at coronary artery",
    "possible infection"
  ]
}
```

<b>processed_annotations</b>

contains info equivalent to this:

<img src="https://i.ibb.co/p2HvjB8/image.png" alt="image" border="0">

<b>radgraph_annotations</b>

output of radgraph engine

<b>all_concepts</b> and <b>concat_concepts</b> will be used for evaluation

Same results can be retrieved with direct annotations (using arg "annotations" from get_concept)

```python
import json

test_annotations = json.load(open("test.radgraph.json"))
key = list(test_annotations.keys())[0]
# creating inference radgraph dict from annotation
single_annotation = test_annotations[key]
new_annotations = {'0': {**single_annotation.pop("labeler_1"), **{'text': single_annotation["text"]}}}
concepts = get_concepts(annotations=new_annotations)
```

#### 2. `get_annotated_text()`
Given annotations of a labeler, get the subtext of the report that contains annotations.
```python
from utils import get_annotated_text

annotations = {'0': {'entities': {'1': {'tokens': 'lungs', ...}
print(new_annotations["0"]["text"])
>> FINAL REPORT INDICATION : ___ F with cough / / Cough TECHNIQUE : PA and lateral views of the chest . COMPARISON : None . FINDINGS : The lungs are clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process .
subtext = get_annotated_text(annotations)
print(subtext)
>> FINDINGS : The lungs are clear without focal consolidation , , or edema . The cardiomediastinal silhouette is within normal limits . No acute osseous abnormalities . IMPRESSION : No acute cardiopulmonary process .
```

### Evaluation
`run.py` evaluate the prediction of the all_concepts and concat_concepts by radgraph against the ground_truth `test.radgraph.json.

right now:
```python
print(f1_score(all_concepts_gt, all_concept_pred, average="micro"))
print(f1_score(concat_concepts_gt, concat_concepts_pred, average="micro"))
print(f1_score(all_concepts_gt, all_concept_pred, average="macro"))
print(f1_score(concat_concepts_gt, concat_concepts_pred, average="macro"))
# 0.851961509992598
# 0.750877192982456
# 0.63318944983074
# 0.5066821072042885
```