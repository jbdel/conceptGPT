import os
from tqdm import tqdm 

import prompt.constants as constants
from report_to_concept.utils import get_concepts, ifcc_clean_report, make_sentences

DATA_DIR = '/home/cvanuden/git-repos/conceptGPT/data/rrs/mimic-cxr'

for split in ['train', 'validate', 'test']:
    findings_list = make_sentences(root=DATA_DIR, split=split, file='findings.tok', processing=ifcc_clean_report)
    findings_list = [" ".join(findings) for findings in findings_list]
    summary_list = make_sentences(root=DATA_DIR, split=split, file='impression.tok', processing=ifcc_clean_report)
    summary_list = [" ".join(summ) for summ in summary_list]

    concepts_list = []
    for finding in tqdm(findings_list, total=len(findings_list)):
        concepts_list.append(get_concepts(report=finding))

    for key in ['all_concepts', 'concat_concepts']:
        concept_path = os.path.join(DATA_DIR, f'{split}.{key}.tok')
        with open(concept_path, 'w') as f:
            for line in concepts_list:
                line_to_write = ','.join(line[key])
                f.write(f"{line_to_write}\n")