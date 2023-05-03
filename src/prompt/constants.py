PROJECT_DIR = '/home/cvanuden/git-repos/conceptGPT'
DATA_DIR = '/home/cvanuden/git-repos/radprompt/data/rrs/mimic-iii'

MODALITY = 'XR_chest'

IN_CONTEXT_LEARNING_SENTENCE_TRANSFORMER = 'pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb'

NUM_BEAMS = 4
MAX_NEW_TOKENS = 128
OPENAI_MAX_CONTEXT_LEN = 4096

HUGGINGFACE_MODELS = {
    'biogpt': 'microsoft/BioGPT',
    'biogpt-large': 'microsoft/BioGPT-Large',
    'biomedlm': 'stanford-crfm/BioMedLM',
    'stablelm-base-alpha-3b': 'stabilityai/stablelm-base-alpha-3b',
    'stablelm-tuned-alpha-3b': 'stabilityai/stablelm-tuned-alpha-3b',
    'stablelm-base-alpha-7b': 'stabilityai/stablelm-base-alpha-7b',
    'stablelm-tuned-alpha-7b': 'stabilityai/stablelm-tuned-alpha-7b',
    'openassistant-stablelm-7b': 'OpenAssistant/stablelm-7b-sft-v7-epoch-3',
    'open-llama': 'openlm-research/open_llama_7b_preview_200bt'
}

OPENAI_MODELS = ['gpt-3.5-turbo', 'gpt-4']

CONFIGS = {
    # 0: {
    #     'start_prefix': "summarize the radiology report \"findings\" into a natural language \"impression\".\n",
    #     'findings_prefix': 'findings:',
    #     'concepts_prefix': 'concepts:',
    #     'prompt': 'impression:',
    #     'concept_type': 'all_concepts',
    #     'use_knn_index': False,
    #     'use_findings': True,
    #     'use_concepts': False,
    # },
    1: {
        'start_prefix': "summarize the radiology report \"findings\" into a natural language \"impression\".\n",
        'findings_prefix': 'findings:',
        'concepts_prefix': 'concepts:',
        'prompt': 'impression:',
        'concept_type': 'all_concepts',
        'use_knn_index': True,
        'use_findings': True,
        'use_concepts': False,
    },
    # 2: {
    #     'start_prefix': "generate a natural language \"impression\". use the provided \"concepts\" for guidance, but do not simply repeat the \"concepts\".\n",
    #     'findings_prefix': 'findings:',
    #     'concepts_prefix': 'concepts:',
    #     'prompt': 'impression:',
    #     'concept_type': 'all_concepts',
    #     'use_knn_index': False,
    #     'use_findings': False,
    #     'use_concepts': True,
    # },
    3: {
        'start_prefix': "generate a natural language \"impression\". use the provided \"concepts\" for guidance, but do not simply repeat the \"concepts\".\n",
        'findings_prefix': 'findings:',
        'concepts_prefix': 'concepts:',
        'prompt': 'impression:',
        'concept_type': 'all_concepts',
        'use_knn_index': True,
        'use_findings': False,
        'use_concepts': True,
    },
    # 4: {
    #     'start_prefix': "summarize the radiology report \"findings\" into a natural language \"impression\". use the provided \"concepts\" for guidance, but do not simply repeat the \"concepts\".\n",
    #     'findings_prefix': 'findings:',
    #     'concepts_prefix': 'concepts:',
    #     'prompt': 'impression:',
    #     'concept_type': 'all_concepts',
    #     'use_knn_index': False,
    #     'use_findings': True,
    #     'use_concepts': True,
    # },
    5: {
        'start_prefix': "summarize the radiology report \"findings\" into a natural language \"impression\". use the provided \"concepts\" for guidance, but do not simply repeat the \"concepts\".\n",
        'findings_prefix': 'findings:',
        'concepts_prefix': 'concepts:',
        'prompt': 'impression:',
        'concept_type': 'all_concepts',
        'use_knn_index': True,
        'use_findings': True,
        'use_concepts': True,
    },
}