PROJECT_DIR = '/home/cvanuden/git-repos/conceptGPT'
DATA_DIR = '/home/cvanuden/git-repos/radprompt/data/rrs/mimic-iii'

IN_CONTEXT_LEARNING_SENTENCE_TRANSFORMER = 'pritamdeka/PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb'

CONFIGS = {
    0: {
        'start_prefix': "summarize the radiology report \"findings\" into an \"impression\".\n",
        'findings_prefix': 'findings:',
        'concepts_prefix': 'concepts:',
        'prompt': 'impression:',
        'k': 2,
        'use_knn_index': False,
        'use_concepts': False,
    },
    1: {
        'start_prefix': "summarize the radiology report \"findings\" into an \"impression\".\n",
        'findings_prefix': 'findings:',
        'concepts_prefix': 'concepts:',
        'prompt': 'impression:',
        'k': 2,
        'use_knn_index': True,
        'use_concepts': False,
    },
    2: {
        'start_prefix': "summarize the radiology report \"findings\" into an \"impression\". use the provided \"concepts\" to help.\n",
        'findings_prefix': 'findings:',
        'concepts_prefix': 'concepts:',
        'prompt': 'impression:',
        'k': 2,
        'use_knn_index': False,
        'use_concepts': True,
    },
    3: {
        'start_prefix': "summarize the radiology report \"findings\" into an \"impression\". use the provided \"concepts\" to help.\n",
        'findings_prefix': 'findings:',
        'concepts_prefix': 'concepts:',
        'prompt': 'impression:',
        'k': 2,
        'use_knn_index': True,
        'use_concepts': True,
    },
}