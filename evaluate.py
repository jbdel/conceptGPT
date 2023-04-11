import os
from radgraph import F1RadGraph
import src.constants


modality = 'CT_chest'
f1radgraph = F1RadGraph(reward_level="partial")

for case in range(1):
    for k in [0, 1, 2]:
        ref_summary_path = os.path.join(src.constants.DATA_DIR, modality, f'test.impression.tok')
        gen_summary_path = os.path.join(src.constants.PROJECT_DIR, f'output/{modality}/case{case}_k{k}_generated_summaries.tok')

        with open(ref_summary_path, 'r') as f_r, open(gen_summary_path, 'r') as f_g:
            ref_summary_list = f_r.readlines()
            gen_summary_list = f_g.readlines()
            
            ref_summary_list = ref_summary_list[:10]
            assert len(ref_summary_list) == len(gen_summary_list)

        radgraph_score, _, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=gen_summary_list, refs=ref_summary_list)

        print(f'CASE: {case}, K: {k}')
        print(f'F1RadGraph score: {radgraph_score}')