import faiss
import os 
import random 
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from nltk.tokenize import wordpunct_tokenize


import prompt.constants as constants

# https://github.com/ysmiura/ifcc/blob/6c111dbdfe7ce9d3150a5ad90360584cfd2b8442/clinicgen/text/tokenizer.py#L24
# Improving Factual Completeness and Consistency of Image-to-text Radiology Report Generation.
def ifcc_clean_report(report):
    report = report.lower()
    return ' '.join(wordpunct_tokenize(report))

def split_sentences(sentences, processing):
    return [processing(s.strip()).split() for s in sentences]


def make_sentences(root, split, file, processing):
    sentences = load_file(os.path.join(root, split + '.' + file))
    return split_sentences(sentences, processing)

def load_file(path):
    """Default loading function, which loads nth sentence at line n.
    """
    with open(path, 'r') as f:
        content = f.read().strip()
    return [s for s in content.split('\n')]


def load_findings_concepts_and_summary(modality, concept_type, split):
    # finding_path = os.path.join(constants.DATA_DIR, modality, f'{split}.findings.tok')
    concept_path = os.path.join(constants.DATA_DIR, modality, f'{split}.{concept_type}.tok')
    # summary_path = os.path.join(constants.DATA_DIR, modality, f'{split}.impression.tok')

    finding_path = os.path.join(f'/home/cvanuden/git-repos/vilmedic/data/RRG/mimic-cxr/findings/{split}.findings.tok')
    summary_path = os.path.join(f'/home/cvanuden/git-repos/vilmedic/data/RRG/mimic-cxr/impression/{split}.impression.tok')

    # with open(finding_path, 'r') as f_f, open(summary_path, 'r') as f_s:
    #     findings_list = f_f.readlines()
    #     summary_list = f_s.readlines()
        
    #     print(len(findings_list), len(summary_list))
    #     assert len(findings_list) == len(summary_list)

    findings_list = make_sentences(root='/home/cvanuden/git-repos/vilmedic/data/RRG/mimic-cxr/findings/', split=split, file='findings.tok', processing=ifcc_clean_report)
    findings_list = [" ".join(findings) for findings in findings_list]
    summary_list = make_sentences(root='/home/cvanuden/git-repos/vilmedic/data/RRG/mimic-cxr/impression/', split=split, file='impression.tok', processing=ifcc_clean_report)
    summary_list = [" ".join(summ) for summ in summary_list]

    print(len(findings_list), len(summary_list), summary_list[0])
    concepts_list = []
    # if os.path.exists(concept_path):
    #     with open(concept_path, 'r') as f_c:
    #         concepts_list = f_c.readlines()

    #         assert len(findings_list) == len(concepts_list)

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
    use_findings=True,
    use_concepts=False,
    sentence_transformer_name=constants.IN_CONTEXT_LEARNING_SENTENCE_TRANSFORMER
):
    ''' create in-context learning prompts '''
    tmp_test_findings_list = []
    tmp_test_summary_list = []

    model = SentenceTransformer(sentence_transformer_name)

    for i, example in tqdm(enumerate(test_findings_list), total=len(test_findings_list)):
        new_example = start_prefix

        # if k == 0, just return the training example with no in-context examples
        if k == 0:
            if use_findings and use_concepts:
                new_example += f"\n\"\"\"\n{findings_prefix} {test_findings_list[i]}\n{concepts_prefix} {test_concepts_list[i]}\n{prompt}"
            elif use_findings:
                new_example += f"\n\"\"\"\n{findings_prefix} {test_findings_list[i]}\n{prompt}"
            elif use_concepts:
                new_example += f"\n\"\"\"\n{concepts_prefix} {test_concepts_list[i]}\n{prompt}"
            else:
                raise ValueError("At least one of use_findings and use_concepts must be true.")

            tmp_test_findings_list.append(new_example)
            tmp_test_summary_list.append(test_summary_list[i])

            continue

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
            # weights = [1. / len(sentence) for sentence in train_findings_list]

            # TODO: samples with replacement - come back to this, may want without
            # example_indices = random.choices(range(len(train_findings_list)), weights=weights, k=k)
            example_indices = random.choices(range(len(train_findings_list)), k=k)

        if use_findings and use_concepts:
            new_example += '\n'.join([f"\"\"\"\n{findings_prefix} {train_findings_list[j]}\n{concepts_prefix} {train_concepts_list[j]}\n{prompt} {train_summary_list[j]}\"\"\"" for j in example_indices])
            new_example += f"\n\"\"\"\n{findings_prefix} {test_findings_list[i]}\n{concepts_prefix} {test_concepts_list[i]}\n{prompt}"
        elif use_findings:
            new_example += '\n'.join([f"\"\"\"\n{findings_prefix} {train_findings_list[j]}\n{prompt} {train_summary_list[j]}\"\"\"" for j in example_indices])
            new_example += f"\n\"\"\"\n{findings_prefix} {test_findings_list[i]}\n{prompt}"
        elif use_concepts:
            new_example += '\n'.join([f"\"\"\"\n{concepts_prefix} {train_concepts_list[j]}\n{prompt} {train_summary_list[j]}\"\"\"" for j in example_indices])
            new_example += f"\n\"\"\"\n{concepts_prefix} {test_concepts_list[i]}\n{prompt}"
        else:
            raise ValueError("At least one of use_findings and use_concepts must be true.")

        tmp_test_findings_list.append(new_example)
        tmp_test_summary_list.append(test_summary_list[i])

    test_findings_list = tmp_test_findings_list
    test_summary_list = tmp_test_summary_list

    return test_findings_list, test_summary_list
