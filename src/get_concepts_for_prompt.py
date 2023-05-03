import os
from tqdm import tqdm 

import prompt.constants as constants
from prompt.utils import load_findings_concepts_and_summary
from report_to_concept.utils import get_concepts

DATA_DIR = '/home/cvanuden/git-repos/vilmedic/data/RRG/mimic-cxr/concepts'

for split in ['train', 'validate', 'test']:
    findings_list, _, summary_list = load_findings_concepts_and_summary(modality=constants.MODALITY, concept_type='all_concepts', split=split)

    concepts_list = []
    for finding in tqdm(findings_list, total=len(findings_list)):
        concepts_list.append(get_concepts(report=finding))

    for key in ['all_concepts', 'concat_concepts']:
        concept_path = os.path.join(DATA_DIR, f'{split}.{key}.tok')
        with open(concept_path, 'w') as f:
            for line in concepts_list:
                line_to_write = ','.join(line[key])
                f.write(f"{line_to_write}\n")