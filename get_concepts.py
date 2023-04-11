import os
from tqdm import tqdm 

import src.constants
from src.prompt.utils import load_findings_concepts_and_summary
from src.report_to_concept.utils import get_concepts

modality = 'XR_chest'

for split in ['train', 'test']:
    findings_list, _, summary_list = load_findings_concepts_and_summary(modality=modality, concept_type='all_concepts', split=split)

    concepts_list = []
    for finding in tqdm(findings_list, total=len(findings_list)):
        concepts_list.append(get_concepts(report=finding))

    for key in ['all_concepts', 'concat_concepts']:
        concept_path = os.path.join(src.constants.DATA_DIR, modality, f'{split}.{key}.tok')
        with open(concept_path, 'w') as f:
            for line in concepts_list:
                line_to_write = ','.join(line[key])
                f.write(f"{line_to_write}\n")