import openai
import backoff  # for exponential backoff

import prompt.constants as constants

openai.organization = "org-4KtF0NDlYTDYngBanKKnzlpd"
openai.api_key = "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"

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def call_openai(prompt, temperature=0, n=1, model="gpt-3.5-turbo"):
    assert model in constants.OPENAI_MODELS, f"`model` must be a valid OpenAI model; choose from {constants.OPENAI_MODELS}"

    if len(prompt) > constants.OPENAI_MAX_CONTEXT_LEN:
        return ''

    return completions_with_backoff(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        n=n,
    )["choices"][0]["message"]["content"]

def call_biogpt_generator(prompt, generator):
    outputs = generator(
        prompt,
        num_beams=constants.NUM_BEAMS,
        max_new_tokens=constants.MAX_NEW_TOKENS,
        early_stopping=True,
        do_sample=False,
        return_full_text=False
    )
    return outputs[0]['generated_text']

def call_lm_generator(prompt, model, tokenizer, device="cuda"):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    tokens = model.generate(
        **inputs,
        num_beams=constants.NUM_BEAMS,
        max_new_tokens=constants.MAX_NEW_TOKENS,
        early_stopping=True,
        do_sample=False,
    )
    return tokenizer.decode(tokens[0], skip_special_tokens=True)
