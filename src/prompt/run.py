import os

import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

import constants
from models import call_biogpt_generator, call_lm_generator, call_openai
from utils import add_in_context_prompt, load_findings_concepts_and_summary, generate_faiss_index

modality = 'XR_chest'
N_EXAMPLES = 10


model_name = 'biomedlm'
case = 3
# set config
CONFIG = constants.CONFIGS[case]

# model setup
device = "cuda"
if 'openai' in model_name:
    pass
if 'biogpt' in model_name:
    generator = pipeline(model=constants.HUGGINGFACE_MODELS[model_name])
else:
    tokenizer = AutoTokenizer.from_pretrained(constants.HUGGINGFACE_MODELS[model_name])
    model = AutoModelForCausalLM.from_pretrained(constants.HUGGINGFACE_MODELS[model_name])
    model.half().cuda()
    


# load data
train_findings_list, train_concepts_list, train_summary_list = load_findings_concepts_and_summary(modality=modality, concept_type=CONFIG['concept_type'], split='train')
test_findings_list, test_concepts_list, test_summary_list = load_findings_concepts_and_summary(modality=modality, concept_type=CONFIG['concept_type'], split='test')

test_findings_list = test_findings_list[:N_EXAMPLES]
test_concepts_list = test_concepts_list[:N_EXAMPLES]
test_summary_list = test_summary_list[:N_EXAMPLES]

# for in-context prompt examples, generate FAISS index with training examples
faiss_index_save_path = os.path.join(constants.PROJECT_DIR, f'data/{modality}/train_findings_index.bin')
if os.path.exists(faiss_index_save_path):
    cpu_index = faiss.read_index(faiss_index_save_path)
else:
    cpu_index = generate_faiss_index(examples=train_findings_list, faiss_index_save_path=faiss_index_save_path)
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

# run all k
for k in [0, 1, 2, 4]:
    case_k_dir = os.path.join(constants.PROJECT_DIR, f'output/{modality}/{model_name}/case{case}/k{k}')
    if not os.path.exists(case_k_dir):
        os.makedirs(case_k_dir)
    findings_path = os.path.join(case_k_dir, 'reference_findings.tok')
    reference_summary_path = os.path.join(case_k_dir, 'reference_summaries.tok')
    generated_summary_path = os.path.join(case_k_dir, 'generated_summaries.tok')
    
    # generate in-context prompts
    in_context_findings_list, in_context_summary_list = add_in_context_prompt(
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
        k=k, 
        knn_index=gpu_index, 
        use_knn_index=CONFIG['use_knn_index'],
        use_findings=CONFIG['use_findings'],
        use_concepts=CONFIG['use_concepts'],
    )

    test_generated_summary_list = []

    if 'openai' not in model_name:
        generator = pipeline(model=constants.HUGGINGFACE_MODELS[model_name])

    for i in tqdm(range(len(in_context_findings_list))):
        prompt = in_context_findings_list[i]
        if 'openai' in model_name:
            # run in-context prompt through openai model
            generated_summary = call_openai(prompt)
        if 'biogpt' in model_name:
            # run in-context prompt through huggingface bioGPT model
            generated_summary = call_biogpt_generator(prompt, generator)
        else:
            # run in-context prompt through huggingface biomedLM or stableLM model
            generated_summary = call_lm_generator(prompt, model, tokenizer)

        test_generated_summary_list.append(generated_summary)

    with open(findings_path, 'w') as f:
        for line in in_context_findings_list:
            # just so it writes a bit cleaner to .tok
            cleaned_line = line.replace('\n', ' ')
            f.write(f"{cleaned_line}\n")
    
    with open(reference_summary_path, 'w') as f:
        for line in in_context_summary_list:
            # just so it writes a bit cleaner to .tok
            cleaned_line = line.replace('\n', ' ')
            f.write(f"{cleaned_line}\n")

    with open(generated_summary_path, 'w') as f:
        for line in test_generated_summary_list:
            # just so it writes a bit cleaner to .tok
            cleaned_line = line.replace('\n', ' ')
            cleaned_line = line.replace('"', '')
            f.write(f"{cleaned_line}\n")