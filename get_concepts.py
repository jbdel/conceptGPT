import os

import src.constants
from src.prompt.utils import load_findings_concepts_and_summary
from src.report_to_concept.utils import get_concepts

modality = 'XR_chest'

for split in ['train', 'test']:
    findings_list, _, summary_list = load_findings_concepts_and_summary(modality=modality, split=split)

    concepts_list = [get_concepts(report=finding) for finding in findings_list]

    concept_path = os.path.join(src.constants.DATA_DIR, modality, f'{split}.concepts.tok')
    with open(concept_path, 'w') as f:
        for line in concepts_list:
            f.write(f"{line}\n")