import json
from typing import Iterable, List, Optional, Tuple
import random
from transformers import PreTrainedTokenizerBase, AutoTokenizer

def sample_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int 
) -> List[Tuple[str, List[int], int, int]]:
    
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    prompts = []
    completions = []
    for i in range(len(dataset)):
        s = ''
        conversations = dataset[i]['conversations']
        if len(conversations) % 2 != 0:
            continue
        for j in range(0, len(conversations), 2):
            human_side = 'Human: ' + conversations[j]['value'] + '\n' 
            gpt_side = 'GPT: ' + conversations[j+1]['value'] + '\n'
            s += human_side
            prompts.append(s)
            completions.append(gpt_side)
            s += gpt_side
    prompt_token_ids = tokenizer(prompts).input_ids
    completion_token_ids = tokenizer(completions).input_ids
    
    reqs = []
    for i in range(len(prompts)):
        prompt_len = len(prompt_token_ids[i])
        completion_len = len(completion_token_ids[i])
        # if prompt_len < 4 or output_len < 4:
        #     # Prune too short sequences.
        #     continue
        # if prompt_len < 2048 or prompt_len + output_len > 4096:
        #     # Prune too long sequences.
        #     continue
        # if prompt_len < 2048 or completion_len < 128 or completion_len > 128 or prompt_len + completion_len > 4096:
        #     # Prune too long sequences.
        #     continue
        reqs.append((prompts[i], prompt_token_ids[i], prompt_len, completion_len))

    print(len(reqs))

    # Sample the requests.
    sampled_requests = random.sample(reqs, num_requests)
    return sampled_requests