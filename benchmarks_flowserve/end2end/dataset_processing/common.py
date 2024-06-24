from typing import Iterable, List, Optional, Tuple

def find_range_of_multi_turn_conversations(requests: List[Tuple[str, List[int], int, int]]) -> List[int]:
    multi_conversations_range = [0]
    first_round_conversation = requests[0]
    for i in range(1, len(requests)):
        n_round_conversation = requests[i]
        if first_round_conversation[0] not in n_round_conversation[0]:
            multi_conversations_range.append(i)
            first_round_conversation = n_round_conversation
    return multi_conversations_range