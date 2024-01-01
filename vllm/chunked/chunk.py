import enum
from typing import List, Dict, Tuple
import torch
from xformers.ops import AttentionBias

from vllm.chunked.chunkcache import Block

class ChunkStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREFILLED = enum.auto()

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
            self.do_cat = False
            for _, prefix in self.seqs_to_prefixs.items():
                if prefix > 0:
                    self.do_cat = True
                    break

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
    
    def set_idxs(self, idxs: List[int]) -> None:
        self.idxs = idxs

    def set_sampling_params_for_sampler(self, sampling_params_for_sampler: List[ChunkSamplingParams]) -> None:
        self.sampling_params_for_sampler = sampling_params_for_sampler

    def set_do_sampling(self, do_sampling: List[str]) -> None:
        self.do_sampling = do_sampling

class Sequence:
    def __init__(self,
                 seq_id: str,
                 prompt_token_ids: List[int],
                 sampling_params: ChunkSamplingParams,
                 account: int = 0,
                 start_time: float = -1.0,
                 end_time: float = -1.0,
                 count: int = 0,
                 processed: bool = False,
                 request_id: List[str] = None,
                 label: int = -1) -> None:
        self.seq_id = seq_id
        self.prompt_token_ids = prompt_token_ids
        self.prompt_len = len(prompt_token_ids)
        self.chunks_to_prompts: Dict[int, int] = {}
        self.sampling_params = sampling_params
        self.outputs: List[torch.Tensor] = []
        self.account = account
        self.start_time = start_time
        self.end_time = end_time
        self.count = count
        self.processed = processed
        self.request_id = request_id
        self.label = label

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
    
    def add_start_and_end_time(self, st: float, ed: float) -> None:
        if self.account == 0:
            self.start_time = st
            self.end_time = ed
            self._update_account(account = 1)
        else:
            self.start_time = min(self.start_time, st)
            self.end_time = max(self.end_time, ed)

    def set_end_time(self, st: float, ed: float) -> None:
        self.start_time = st
        self.end_time = ed

    def _update_account(self, account: int) -> None:
        self.account = account
    
    def add_sampler_time(self, st: float, ed: float) -> None:
        self.sampler_start = st
        self.sampler_end = ed

    def update_count(self, input: int) -> None:
        self.count += input
    
    def is_full(self) -> bool:
        if self.count == self.prompt_len:
            return True
        else:
            return False

class ChunkInputMetadata:
    def __init__(self,
                 prompt_lens: List[int],
                 kv_prefixs: List[int],
                 kv_prefixs_blocks: Dict[int, List[Tuple[int, int, int]]],
                 kv_block: int,
                 idxs: List[int],
                 sampling_params_for_sampler: List[ChunkSamplingParams],
                 do_cat: bool) -> None:
        self.prompt_lens = prompt_lens
        self.kv_prefixs = kv_prefixs
        self.kv_prefixs_blocks = kv_prefixs_blocks
        self.kv_block = kv_block
        self.valid_tokens_num = sum(prompt_lens)
        self.attn_bias: List[AttentionBias] = []
        self.idxs = idxs
        self.sampling_params_for_sampler = sampling_params_for_sampler
        self.do_cat = do_cat