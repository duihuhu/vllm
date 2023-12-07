import enum
from typing import Optional, List, Dict, Tuple
import torch
from xformers.ops import AttentionBias

from vllm.chunked.chunkcache import Block

class ChunkStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREFILLED = enum.auto()

class Chunk:
    def __init__(self,
                 chunk_id: int,
                 chunk_size: int,
                 chunk_status: ChunkStatus) -> None:
        self.chunk_id = chunk_id
        self.chunk_size = chunk_size
        self.chunk_status = chunk_status
        self.prompt_lens: List[int] = []
        self.prompt_token_ids: List[int] = []
        self.kv_prefixs: List[int] = []
        self.raw_sequence_ids: List[str] = []
    
    def set_seqs_to_lens_and_prefixs(self) -> None:
        if len(self.raw_sequence_ids) > 0 and len(self.prompt_lens) > 0 and len(self.kv_prefixs) > 0:
            self.seqs_to_lens = self._set_seqs_to_lens()
            self.seqs_to_prefixs = self._set_seqs_to_prefixs()

    def _set_seqs_to_lens(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i, seq_id in enumerate(self.raw_sequence_ids):
            out[seq_id] = self.prompt_lens[i]
        return out

    def _set_seqs_to_prefixs(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for i, seq_id in enumerate(self.raw_sequence_ids):
            out[seq_id] = self.kv_prefixs[i]
        return out
    
    def set_self_block(self, block: Block) -> None:
        self.cache_block = block
        self.cache_block_id = block.block_id

class ChunkSamplingParams:
    def __init__(self,
                 temperature: float,
                 top_p: float,
                 top_k: int) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_beam_search = False
        self.best_of = 1

class Sequence:
    def __init__(self,
                 seq_id: str,
                 prompt: Optional[str],
                 prompt_token_ids: List[int],
                 sampling_params: ChunkSamplingParams) -> None:
        self.seq_id = seq_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_len = len(prompt_token_ids)
        self.chunks_to_prompts: Dict[int, int] = {}
        self.sampling_params = sampling_params
        self.outputs: List[torch.Tensor] = []
    
    def append_outputs(self, input: torch.Tensor) -> None:
        self.outputs.append(input)

    def get_output_tensor(self) -> torch.Tensor:
        out = self.outputs[0]
        length = len(self.outputs)
        for i in range(1, length):
            out = torch.cat((out, self.outputs[i]), 0)
        return out

    def add_first_token_id(self, new_token_id: int) -> None:
        self.first_token_id = new_token_id
    
    def add_first_token_logprob(self, logprob: float) -> None:
        self.first_token_logprob = logprob
    
    def add_first_token_str(self, new_output_text: str) -> None:
        self.first_token_str = new_output_text

class ChunkInputMetadata:
    def __init__(self,
                 prompt_lens: List[int],
                 kv_prefixs: List[int],
                 kv_prefixs_blocks: Dict[int, Tuple[int, int, int]],
                 kv_block: int) -> None:
        self.prompt_lens = prompt_lens
        self.kv_prefixs = kv_prefixs
        self.kv_prefixs_blocks = kv_prefixs_blocks
        self.kv_block = kv_block
        self.valid_tokens_num = sum(prompt_lens)
        self.attn_bias: List[AttentionBias] = []