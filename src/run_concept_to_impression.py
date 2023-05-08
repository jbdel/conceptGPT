import os

import argparse
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

import prompt.constants as constants
from prompt.models import call_biogpt_generator, call_lm_generator, call_openai
from prompt.utils import add_in_context_prompt, load_data, generate_faiss_index

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='biogpt', help="Model to use for concept->impression prompting.")
    parser.add_argument('--case_id', type=int, default=3, help="How to prompt the model.")
    parser.add_argument('--modality', type=str, default='XR_chest', help="Modality to prompt for.")
    parser.add_argument('--n_examples', type=int, default=None, help="How many examples to prompt for.")
    parser.add_argument('--in_context_k', type=int, default=0, help="How many in-context examples to provide in prompt.")
    parser.add_argument('--run_in_context_k_sweep', action='store_true', help="Whether to run a sweep (powers of 2) up to in_context_k.")
    args = parser.parse_args()
    return args


def prompt_pipeline(args):
    # set config
    config = constants.CONFIGS[args.case_id]

    # model setup
    device = "cuda"
    if 'openai' in args.model_name:
        pass
    if 'biogpt' in args.model_name:
        generator = pipeline(model=constants.HUGGINGFACE_MODELS[args.model_name])
    else:
        tokenizer = AutoTokenizer.from_pretrained(constants.HUGGINGFACE_MODELS[args.model_name])
        model = AutoModelForCausalLM.from_pretrained(constants.HUGGINGFACE_MODELS[args.model_name])
        model.half().cuda()

    # load data
    train_findings_list = load_data(modality=args.modality, data_type='findings', split='train')
    train_concepts_list = load_data(modality=args.modality, data_type=config['concept_type'], split='train')
    train_summary_list = load_data(modality=args.modality, data_type='impression', split='train')

    test_findings_list = load_data(modality=args.modality, data_type='findings', split='test')
    test_concepts_list = load_data(modality=args.modality, data_type=config['concept_type'], split='test')
    test_summary_list = load_data(modality=args.modality, data_type='impression', split='test')

    if args.n_examples is not None:
        test_findings_list = test_findings_list[:args.n_examples]
        test_concepts_list = test_concepts_list[:args.n_examples]
        test_summary_list = test_summary_list[:args.n_examples]

    # for in-context prompt examples, generate FAISS index with training examples
    faiss_index_save_path = os.path.join(constants.PROJECT_DIR, f'data/{args.modality}/train_findings_index.bin')
    if os.path.exists(faiss_index_save_path):
        cpu_index = faiss.read_index(faiss_index_save_path)
    else:
        cpu_index = generate_faiss_index(examples=train_findings_list, faiss_index_save_path=faiss_index_save_path)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)

    # create in-context k
    if args.run_in_context_k_sweep:
        # power of 2 sweep
        k = 0
        in_context_k = [k]
        exponent = 0
        while k < args.in_context_k:
            k = 1<<exponent
            in_context_k.append(k)
            exponent += 1
    else:
        in_context_k = [args.in_context_k]

    # run all k
    for k in in_context_k:
        case_k_dir = os.path.join(constants.PROJECT_DIR, f'output/{args.modality}/{args.model_name}/case{args.case_id}/k{k}')
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
            start_prefix=config['start_prefix'],
            findings_prefix=config['findings_prefix'],
            concepts_prefix=config['concepts_prefix'],
            prompt=config['prompt'],
            k=k, 
            knn_index=gpu_index, 
            use_knn_index=config['use_knn_index'],
            use_findings=config['use_findings'],
            use_concepts=config['use_concepts'],
        )

        test_generated_summary_list = []

        if 'openai' not in args.model_name:
            generator = pipeline(model=constants.HUGGINGFACE_MODELS[args.model_name])

        for i in tqdm(range(len(in_context_findings_list))):
            prompt = in_context_findings_list[i]
            if 'openai' in args.model_name:
                # run in-context prompt through openai model
                generated_summary = call_openai(prompt)
            if 'biogpt' in args.model_name:
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

if __name__ == "__main__":
    args = parse_args()
    model = prompt_pipeline(args)