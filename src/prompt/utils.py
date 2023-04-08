import faiss
import os 
import random 
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import src.constants as constants

def load_findings_concepts_and_summary(modality, split):
    finding_path = os.path.join(constants.DATA_DIR, modality, f'{split}.findings.tok')
    concept_path = os.path.join(constants.DATA_DIR, modality, f'{split}.concepts.tok')
    summary_path = os.path.join(constants.DATA_DIR, modality, f'{split}.impression.tok')

    with open(finding_path, 'r') as f_f, open(summary_path, 'r') as f_s:
        findings_list = f_f.readlines()
        
        summary_list = f_s.readlines()
        
        assert len(findings_list) == len(summary_list)

    concepts_list = []
    if os.path.exists(concept_path):
        with open(concept_path, 'r') as f_c:
            concepts_list = f_c.readlines()

            len(findings_list) == len(concepts_list)

    return findings_list, concepts_list, summary_list


def generate_faiss_index(examples, faiss_index_save_path, sentence_transformer_name=constants.IN_CONTEXT_LEARNING_SENTENCE_TRANSFORMER):
    model = SentenceTransformer(sentence_transformer_name)

    # compute x_index
    x_index = model.encode(examples, convert_to_tensor=True)

    # train the FAISS index
    # this is helpful: https://github.com/matsui528/faiss_tips
    D = x_index.shape[1]
    cpu_index = faiss.IndexFlatL2(D)
    knn_index = faiss.index_cpu_to_all_gpus(cpu_index)
    knn_index.add(x_index.cpu())
    cpu_index = faiss.index_gpu_to_cpu(knn_index)

    # and save the index
    faiss.write_index(cpu_index, faiss_index_save_path)

    return cpu_index


def add_in_context_prompt(
    train_findings_list, 
    train_concepts_list,
    train_summary_list, 
    test_findings_list, 
    test_concepts_list,
    test_summary_list, 
    split, 
    start_prefix='',
    findings_prefix='',
    concepts_prefix='',
    prompt='',
    k=0, 
    knn_index=None,
    use_knn_index=False,
    use_concepts=False,
    sentence_transformer_name=constants.IN_CONTEXT_LEARNING_SENTENCE_TRANSFORMER
):
    ''' create in-context learning prompts '''
    tmp_test_findings_list = []
    tmp_test_summary_list = []

    model = SentenceTransformer(sentence_transformer_name)

    for i, example in tqdm(enumerate(test_findings_list), total=len(test_findings_list)):
        new_example = start_prefix

        # now add the in-context examples
        if use_knn_index:
            # fetch the kNN from our pretrained FAISS index
            # since if training, the train example is included in the trained index
            k_query = k + 1 if split=='train' else k
            x_query = model.encode(example, convert_to_tensor=True)
            x_query = x_query.unsqueeze(0).cpu()
            example_dists, example_indices = knn_index.search(x=x_query, k=k_query)
            example_indices = example_indices[0]  # unnest single query
            example_indices = example_indices[1:] if split=='train' else example_indices
            example_indices = example_indices.tolist()
        else:
            # random sample, but weight towards shorter examples
            # s.t. it's more likely we can fit whole prompt in context length
            weights = [1. / len(sentence) for sentence in train_findings_list]

            # TODO: samples with replacement - come back to this, may want without
            example_indices = random.choices(range(len(train_findings_list)), weights=weights, k=k)

        if use_concepts:
            new_example += '\n'.join([f"\"\"\"\n{findings_prefix} {train_findings_list[j]}\n{concepts_prefix} {train_concepts_list[j]}\n{prompt} {train_summary_list[j]}\"\"\"" for j in example_indices])
            new_example += f"\n\"\"\"\n{findings_prefix} {test_findings_list[i]}\n{concepts_prefix} {test_concepts_list[j]}\n{prompt}"
        else:
            new_example += '\n'.join([f"\"\"\"\n{findings_prefix} {train_findings_list[j]}\n{prompt} {train_summary_list[j]}\"\"\"" for j in example_indices])
            new_example += f"\n\"\"\"\n{findings_prefix} {test_findings_list[i]}\n{prompt}"

        tmp_test_findings_list.append(new_example)
        tmp_test_summary_list.append(test_summary_list[i])

    test_findings_list = tmp_test_findings_list
    test_summary_list = tmp_test_summary_list

    return test_findings_list, test_summary_list
