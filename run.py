import os

import faiss
from sentence_transformers import SentenceTransformer

import src.constants
from src.prompt.utils import add_in_context_prompt, load_findings_concepts_and_summary, generate_faiss_index
from src.report_to_concept.utils import call_chatgpt

modality = 'XR_chest'

# set config
CONFIG = src.constants.CONFIGS[1]
print('CONFIG:')
print(CONFIG)

train_findings_list, train_concepts_list, train_summary_list = load_findings_concepts_and_summary(modality=modality, split='train')
test_findings_list, test_concepts_list, test_summary_list = load_findings_concepts_and_summary(modality=modality, split='test')

test_findings_list = test_findings_list[:10]
test_concepts_list = test_concepts_list[:10]
test_summary_list = test_summary_list[:10]

# for in-context prompt examples, generate FAISS index with training examples
faiss_index_save_path = os.path.join(src.constants.PROJECT_DIR, f'data/{modality}/train_findings_index.bin')
if os.path.exists(faiss_index_save_path):
    cpu_index = faiss.read_index(faiss_index_save_path)
else:
    cpu_index = generate_faiss_index(examples=train_findings_list, faiss_index_save_path=faiss_index_save_path)
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

# generate in-context prompts
test_findings_list, test_summary_list = add_in_context_prompt(
    train_findings_list,
    train_concepts_list,
    train_summary_list, 
    test_findings_list, 
    test_concepts_list,
    test_summary_list, 
    split='test', 
    start_prefix=CONFIG['start_prefix'],
    findings_prefix=CONFIG['findings_prefix'],
    concepts_prefix=CONFIG['concepts_prefix'],
    prompt=CONFIG['prompt'],
    k=CONFIG['k'], 
    knn_index=gpu_index, 
    use_knn_index=CONFIG['use_knn_index'],
    use_concepts=CONFIG['use_concepts'],
)

in_context_prompt = test_findings_list[0]
ref_summary = test_summary_list[0]

# run in-context prompt through gpt
generated_summary = call_chatgpt(
        prompt=in_context_prompt,
        temperature=0,
        n=1)["choices"][0]["message"]["content"]

print('PROMPT:')
print(in_context_prompt)
print('REFERENCE:')
print(ref_summary)
print('GENERATED:')
print(generated_summary)

# from radgraph import F1RadGraph
# f1radgraph = F1RadGraph(reward_level="partial")
# radgraph_score, _, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=[generated_summary], refs=[ref_summary])

# print(f'F1RadGraph score: {radgraph_score}')