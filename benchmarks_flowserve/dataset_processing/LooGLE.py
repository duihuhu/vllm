import json
from typing import Iterable, List, Optional, Tuple
import random
from transformers import PreTrainedTokenizerBase
from utils import find_range_of_multi_turn_conversations

def sample_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int 
) -> List[Tuple[str, List[int], int, int]]:
    
    doc_QAs = []
    for file_path in dataset_path:
        with open(file_path, 'r') as f:
            for line in f:
                doc_QAs.append(json.loads(line)) 
        
    prompts = []
    completions = []
    for i in range(len(doc_QAs)):
        doc = 'Title: ' + doc_QAs[i]['title'] + '\n' + doc_QAs[i]['input']
        QAs = eval(doc_QAs[i]['qa_pairs'])
        for j in range(len(QAs)):
            question = QAs[j]['Q']
            answer = QAs[j]['A']
            prompts.append(doc + '\nQuestion: ' + question)
            completions.append(answer)

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

    # print(len(reqs))

    # Sample the requests.
    sampled_requests = reqs[:num_requests]
    
    multi_conversations_range = find_range_of_multi_turn_conversations(sampled_requests)
    
    return sampled_requests, multi_conversations_range