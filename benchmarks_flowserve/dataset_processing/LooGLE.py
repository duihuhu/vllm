import json
from typing import Iterable, List, Optional, Tuple
import random
from transformers import PreTrainedTokenizerBase
import pickle 
import os
from .common import find_range_of_multi_turn_conversations
from .ReAct import PROMPT_FORWORD

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
        doc_QAs = []
        with open(dataset_path, 'r') as f:
            for line in f:
                doc_QAs.append(json.loads(line)) 
            
        prompts = []
        completions = []
        for i in range(len(doc_QAs)):
            # Cut the long document
            s = 'Title: ' + doc_QAs[i]['title'] + '\n' + doc_QAs[i]['input'][:len(PROMPT_FORWORD)] + '\n'
            QAs = eval(doc_QAs[i]['qa_pairs'])
            for j in range(len(QAs)):
                # Since there are too many QAs for one single document, we only sample the first 5 QAs
                if j > 5:
                    break
                s += 'Question: ' + QAs[j]['Q'] + '\n'
                prompts.append(s)
                completions.append('Answer: ' + QAs[j]['A'] + '\n')
                s += completions[-1]

        prompt_token_ids = tokenizer(prompts).input_ids
        completion_token_ids = tokenizer(completions).input_ids

        reqs = []
        for i in range(len(prompts)):
            prompt_len = len(prompt_token_ids[i])
            completion_len = len(completion_token_ids[i])
            if prompt_len + completion_len > 4090:
                continue
            reqs.append((prompts[i], prompt_token_ids[i], prompt_len, completion_len))
        # Caution: we only cache the first 1024 requests
        pickle.dump(reqs[:1024], open(cached_file_name, 'wb')) 

    sampled_requests = reqs[:num_requests]
    while len(sampled_requests) < num_requests:
        sampled_requests.extend(reqs[:num_requests - len(sampled_requests)])
    multi_conversations_range = find_range_of_multi_turn_conversations(sampled_requests)
    
    return sampled_requests, multi_conversations_range