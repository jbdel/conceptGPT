
`get_concept()` in report_to_concepts/utils.py

```python
import json
reports = [
    "1 . nodular opacities in the left lower lung with additional small ground-glass opacities bilaterally may represent infection . chest ct recommended for further assessment given infectious symptoms . 2 . abdominal wall varices of indeterminate etiology . 3 . splenomegaly . 4 . coronary artery calcification . acute findings were discussed with dr . ___ by dr . ___ by telephone at 6 : 54 p . m . on ___ ."
]
concepts = get_concepts(report=reports[0])
print(json.dumps(concepts, indent=4))
```
outputs
```json
{
    "processed_annotations": [...],
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
Not very useful for this project

<b>radgraph_annotations</b>

output of radgraph engine

<b>all_concepts</b> and <b>concat_concepts</b> will be used for evaluation

Same results can be retrieved with direct radgraph annotations:

```python
import json
test_annotations = json.load(open("chest-xray-testset.json"))
key = list(test_annotations.keys())[0]
# creating inference radgraph dict from annotation
single_annotation = test_annotations[key]
new_annotations = {'0': {**single_annotation.pop("labeler_1"), **{'text': single_annotation["text"]}}}
concepts = get_concepts(annotations=new_annotations)
```
