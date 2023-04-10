import openai
import backoff  # for exponential backoff

from collections import defaultdict

openai.organization = "org-4KtF0NDlYTDYngBanKKnzlpd"
openai.api_key = "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"


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def call_chatgpt(prompt, temperature, n):
    return completions_with_backoff(
        # model="gpt-4",
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        n=n,
    )


def correct_modifier_order(modifiers):
    prompt = "Please write only the answer to the question and nothing else (no justification, no explanations, ...).\n" \
             "The question is: what would be the most common natural word order for the words: '{}'?\n" \
             "If the word order is correct already, just write the words again.".format(modifiers)
    modifiers = call_chatgpt(
        prompt=prompt,
        temperature=0,
        n=1)["choices"][0]["message"]["content"]

    return modifiers


def recursive_modifier(annotations, ind, modify_dict, done=None):
    if done is None:
        done = set()

    if ind in done:
        return []
    done.add(ind)

    modifiers = []
    if ind in modify_dict:
        for modifier_index in modify_dict[ind]:
            modifiers += recursive_modifier(annotations, modifier_index, modify_dict, done)
    modifiers.append((annotations[ind]["tokens"], annotations[ind]["label"], annotations[ind]["start_ix"]))
    return modifiers


def sort_words_by_index(word_list, index_list):
    sorted_words = [word_list[index_list.index(i)] for i in sorted(index_list)]
    return sorted_words


def get_radgraph_processed_annotations(radgraph_annotations):
    annotations: dict = radgraph_annotations["0"]["entities"]
    radgraph_text = radgraph_annotations["0"]["text"]

    obs_modified_by_obs = defaultdict(list)
    obs_located_anat = defaultdict(list)
    obs_suggest_obs = defaultdict(list)
    anat_modify_anat = defaultdict(list)
    all_observations = []

    tags_name = {
        "OBS-U": 'uncertain',
        "OBS-DP": 'definitely present',
        "OBS-DA": 'definitely absent',
    }

    # First loop over all entities
    for index, v in annotations.items():
        label = v["label"]
        tokens = v["tokens"]
        relations = v["relations"]

        # We consider a "main observation" an OBS entity that does not modify
        if 'OBS' in label and not ('modify' in [relation[0] for relation in relations]):
            all_observations.append((index, tokens, label))

        # For the current entity, fill the following dict
        for relation in relations:
            if 'modify' in relation and 'OBS' in label:
                target = relation[1]
                obs_modified_by_obs[target].append(index)

            if 'modify' in relation and 'ANAT' in label:
                target = relation[1]
                anat_modify_anat[target].append(index)

            if 'located_at' in relation and 'OBS' in label:
                target = relation[1]
                obs_located_anat[index].append(target)

            if 'suggestive_of' in relation and 'OBS' in label:
                target = relation[1]
                obs_suggest_obs[index].append(target)

    processed_observations = []
    # For each main observation
    for observation in all_observations:
        record = {
            "observation": None,
            "observation_start_ix": [],
            "located_at": [],
            "located_at_start_ix": [],
            "tags": [],
            "suggestive_of": []
        }
        observation_index, tokens, label = observation

        # Recursively get full observation name with modifiers (such as increased)
        # We also return the labels, and start_ix (index in the sentence) of retrieved entities
        modifiers = recursive_modifier(annotations, observation_index, obs_modified_by_obs)
        modifiers_labels = [m[1] for m in modifiers]
        modifiers_start_ix = [int(m[2]) for m in modifiers]
        modifiers_tokens = [m[0] for m in modifiers]

        # We rearrange the words according to start_ix (order they appear in the sentence)
        # for example wall abdominal -> abdominal wall
        modifiers_tokens = sort_words_by_index(modifiers_tokens, modifiers_start_ix)
        modifiers_start_ix = sorted(modifiers_start_ix)

        # Sometimes, because of recursivity, we fetch twice the same modifiers (two modifiers have the same modifiers)
        # Need to filter
        modifiers_tokens = [x for i, x in enumerate(modifiers_tokens) if x not in modifiers_tokens[:i]]
        modifiers_start_ix = [x for i, x in enumerate(modifiers_start_ix) if
                              x not in modifiers_start_ix[:i]]

        record["observation"] = " ".join(modifiers_tokens).lower().strip("\n .\"'")
        record["observation_start_ix"] = modifiers_start_ix

        # We prepend "uncertain" or "no" to the full observation if the obs or modifiers contain the corresponding label
        # This is working well.
        if "OBS-DA" in modifiers_labels:
            record["observation"] = "no " + record["observation"]
            tag = tags_name["OBS-DA"]
        elif "OBS-U" in modifiers_labels:
            record["observation"] = "possible " + record["observation"]
            tag = tags_name["OBS-U"]
        else:
            tag = tags_name["OBS-DP"]

        # Tag
        record["tags"] = [tag]

        # We do the exact same for the anatomies of main observations (in obs_located_anat)
        if observation_index in obs_located_anat:

            located_at = []
            located_at_start_ix = []
            anats_index = obs_located_anat[observation_index]

            # recursively retrieve modifiers
            for anat_index in anats_index:
                modifiers = recursive_modifier(annotations, anat_index, anat_modify_anat)
                modifiers_start_ix = [int(m[2]) for m in modifiers]
                modifiers_tokens = [m[0] for m in modifiers]

                # Sorting
                modifiers_tokens = sort_words_by_index(modifiers_tokens, modifiers_start_ix)
                modifiers_start_ix = sorted(modifiers_start_ix)

                # No duplicates
                modifiers_tokens = [x for i, x in enumerate(modifiers_tokens) if x not in modifiers_tokens[:i]]
                modifiers_start_ix = [x for i, x in enumerate(modifiers_start_ix) if
                                      x not in modifiers_start_ix[:i]]

                located_at.append(" ".join(modifiers_tokens).lower().strip("\n .\"'"))
                located_at_start_ix.append(modifiers_start_ix)

            record["located_at"] = located_at
            record["located_at_start_ix"] = located_at_start_ix

        # Suggestive of
        if observation_index in obs_suggest_obs:
            targets = obs_suggest_obs[observation_index]
            suggestive_of_records = []
            for target in targets:
                suggestive_of_records.append(tokens + " suggestive of " + annotations[target]["tokens"])

            record["suggestive_of"] = suggestive_of_records

        processed_observations.append(record)

    return {"processed_annotations": processed_observations,
            "radgraph_annotations": radgraph_annotations,
            "radgraph_text": radgraph_text}


def get_annotated_text(new_annotations):
    entities = new_annotations["0"]["entities"].values()

    start_ix = min(entity["start_ix"] for entity in entities)
    end_ix = max(entity["end_ix"] for entity in entities)

    subtext = new_annotations["0"]["text"].split(" ")

    while start_ix > 0 and subtext[start_ix - 1] != ".":
        start_ix -= 1

    while end_ix < len(subtext) - 1 and subtext[end_ix] != ".":
        end_ix += 1

    subtext = " ".join(subtext[start_ix:end_ix + 1]).strip()
    return subtext


def annotations_to_concepts(annotations):
    concepts = []
    observation = []
    located_at = []
    suggestive_of = []

    annotations = get_radgraph_processed_annotations(annotations)
    # Unrolling all concepts. I.e. observation, located_at, suggestive_of are distinct concepts
    for annotation in annotations["processed_annotations"]:
        for key in ["observation", "located_at", "suggestive_of"]:
            if isinstance(annotation[key], list):
                for elem in annotation[key]:
                    concepts.append(elem)
            else:
                concepts.append(annotation[key])

    annotations["all_concepts"] = list(set(concepts))

    # concatenated concepts
    concepts = []
    for annotation in annotations["processed_annotations"]:
        obs = annotation["observation"]
        located_at = " and ".join(annotation["located_at"])
        suggestive_of = " and ".join([elem[elem.index("of") + 3:] for elem in annotation["suggestive_of"]])
        if located_at:
            obs = obs + " located at " + located_at
        if suggestive_of:
            obs = obs + " suggestive of " + suggestive_of
        concepts.append(obs)

    annotations["concat_concepts"] = list(set(concepts))
    return annotations
