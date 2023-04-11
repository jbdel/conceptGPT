import os

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import src.constants
from src.prompt.utils import add_in_context_prompt, call_chatgpt, load_findings_concepts_and_summary, generate_faiss_index
# from src.report_to_concept.utils import call_chatgpt

modality = 'XR_chest'
N_EXAMPLES = 10

for case in range(6):
    # set config
    CONFIG = src.constants.CONFIGS[case]

    # load data
    train_findings_list, train_concepts_list, train_summary_list = load_findings_concepts_and_summary(modality=modality, concept_type=CONFIG['concept_type'], split='train')
    test_findings_list, test_concepts_list, test_summary_list = load_findings_concepts_and_summary(modality=modality, concept_type=CONFIG['concept_type'], split='test')

    test_findings_list = test_findings_list[:N_EXAMPLES]
    test_concepts_list = test_concepts_list[:N_EXAMPLES]
    test_summary_list = test_summary_list[:N_EXAMPLES]

    # for in-context prompt examples, generate FAISS index with training examples
    faiss_index_save_path = os.path.join(src.constants.PROJECT_DIR, f'data/{modality}/train_findings_index.bin')
    if os.path.exists(faiss_index_save_path):
        cpu_index = faiss.read_index(faiss_index_save_path)
    else:
        cpu_index = generate_faiss_index(examples=train_findings_list, faiss_index_save_path=faiss_index_save_path)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

    # run all k
    for k in [0, 1, 2, 4, 8]:
        generated_summary_path = os.path.join(src.constants.PROJECT_DIR, f'output/{modality}/case{case}_k{k}_generated_summaries.tok')

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
            k=k, 
            knn_index=gpu_index, 
            use_knn_index=CONFIG['use_knn_index'],
            use_findings=CONFIG['use_findings'],
            use_concepts=CONFIG['use_concepts'],
        )

        test_generated_summary_list = []

        n_too_long = 0
        for i in tqdm(range(len(test_findings_list))):
            # prompt too long!
            if len(test_findings_list[i]) > 4096:
                test_generated_summary_list.append('')
                n_too_long += 1
            else:
                # run in-context prompt through gpt
                generated_summary = call_chatgpt(
                        prompt=test_findings_list[i],
                        temperature=0,
                        n=1)["choices"][0]["message"]["content"]

                # just so it writes a bit cleaner to .tok
                generated_summary = generated_summary.replace('\n', ' ')
                test_generated_summary_list.append(generated_summary)

        print(f'CASE: {case}, k: {k}, N_TOO_LONG: {n_too_long}')
        # print('PROMPT:')
        # print(test_findings_list[0]+'\n')
        # print('REFERENCE:')
        # print(test_summary_list[0])
        # print('GENERATED:')
        # print(test_generated_summary_list[0])

        with open(generated_summary_path, 'w') as f:
            for line in test_generated_summary_list:
                f.write(f"{line}\n")