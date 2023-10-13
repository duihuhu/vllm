"""Token blocks."""
from typing import List, Optional, Tuple

from vllm.utils import Device
import pyarrow._plasma as plasma_object

_BLANK_TOKEN_ID = -1

PlasmaObjectIDS = List[Tuple[plasma_object.ObjectID, plasma_object.ObjectID]]

class LogicalTokenBlock:
    """A block that stores a contiguous chunk of tokens from left to right.

    Logical blocks are used to represent the states of the corresponding
    physical blocks in the KV cache.
    """

    def __init__(
        self,
        block_number: int,
        block_size: int,
    ) -> None:
        self.block_number = block_number
        self.block_size = block_size

        self.token_ids = [_BLANK_TOKEN_ID] * block_size
        self.num_tokens = 0

    def is_empty(self) -> bool:
        return self.num_tokens == 0

    def get_num_empty_slots(self) -> int:
        return self.block_size - self.num_tokens

    def is_full(self) -> bool:
        return self.num_tokens == self.block_size

    def append_tokens(self, token_ids: List[int]) -> None:
        assert len(token_ids) <= self.get_num_empty_slots()
        curr_idx = self.num_tokens
        self.token_ids[curr_idx:curr_idx + len(token_ids)] = token_ids
        self.num_tokens += len(token_ids)

    def get_token_ids(self) -> List[int]:
        return self.token_ids[:self.num_tokens]

    def get_last_token_id(self) -> int:
        assert self.num_tokens > 0
        return self.token_ids[self.num_tokens - 1]

class PhysicalTokenBlock:
    """Represents the state of a block in the KV cache."""

    def __init__(
        self,
        device: Device,
        block_number: int,
        block_size: int,
        # object_id: Optional[List[plasma_object.ObjectID]] = None,
        num_layer_object: Optional[int] = None,
        plasma_objects_ids: Optional[List[PlasmaObjectIDS]] = None
    ) -> None:
        self.device = device
        self.block_number = block_number
        self.block_size = block_size

        self.ref_count = 0
        #to reprsent plasma object_id    
        # self.object_id = object_id
        
        #to reprsent to worker: how many layers
        self.num_layer_object = num_layer_object
        #to reprsent a id list which in shape of [worker_num...[num_layers...]]
        self.plasma_objects_ids = plasma_objects_ids
    def __repr__(self) -> str:
        return (f'PhysicalTokenBlock(device={self.device}, '
                f'block_number={self.block_number}, '
                f'ref_count={self.ref_count})')
