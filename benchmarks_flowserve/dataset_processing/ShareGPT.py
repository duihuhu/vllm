import json
from typing import Iterable, List, Optional, Tuple
import random
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import os
import pickle
from .common import find_range_of_multi_turn_conversations

def sample_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int 
) -> Tuple[List[Tuple[str, List[int], int, int]], List[int]]:
    
    cached_file_name = dataset_path.split('.')[0] + '_cached.pkl'
    if os.path.exists(cached_file_name):
        with open(cached_file_name, 'rb') as f:
            reqs = pickle.load(f)
    else:
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
            if prompt_len + completion_len > 4090:
                continue
            reqs.append((prompts[i], prompt_token_ids[i], prompt_len, completion_len))
        # Caution: we only cache the first 512 requests
        pickle.dump(reqs[:512], open(cached_file_name, 'wb'))

    sampled_requests = reqs[:num_requests]
    while len(sampled_requests) < num_requests:
        sample_requests.extend(reqs[:num_requests - len(sampled_requests)])
    multi_conversations_range = find_range_of_multi_turn_conversations(sampled_requests)
    
    return sampled_requests, multi_conversations_range