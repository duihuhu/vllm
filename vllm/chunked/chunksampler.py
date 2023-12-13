"""A layer that samples the next tokens from the model's outputs."""
from typing import List, Tuple, Optional#, Dict 

#import numpy as np
import torch
import torch.nn as nn

#from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.parallel_utils.tensor_parallel import (
    gather_from_tensor_model_parallel_region)
#from vllm.sampling_params import SamplingParams
#from vllm.sequence import SequenceOutputs
from vllm.chunked.chunk import ChunkSamplingParams

_SAMPLING_EPS = 1e-5

class ChunkSampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    1. Discard the hidden states that are not used for sampling (i.e., all
        tokens except the final one in each prompt).
    2. Compute the logits for the next tokens.
    3. Apply presence and frequency penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        embedding: torch.Tensor,
        hidden_states: torch.Tensor,
        #input_metadata: InputMetadata,
        sampling_params: List[ChunkSamplingParams],
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[List[int], List[float]]:
        # Get the hidden states that we use for sampling.
        # sync the outputs among different chunks and then
        # generate the new token
        #hidden_states = hidden_states[-1]
        # reshape for match later
        if len(hidden_states.shape) == 1:
            hidden_states = hidden_states.reshape(1, -1)

        # Get the logits for the next tokens.
        logits = torch.matmul(hidden_states, embedding.t())
        if embedding_bias is not None:
            logits += embedding_bias
        # temporarily this is useless due to the tp==1   
        logits = gather_from_tensor_model_parallel_region(logits)
        # Remove paddings in vocab (if any).
        logits = logits[:, :self.vocab_size]

        # Apply presence and frequency penalties.
        # Only process prefill, so delete all codes for process prefill&decode       

        # Apply temperature scaling.
        temperatures = _get_temperature(sampling_params)
        if any(temperature != 1.0 for temperature in temperatures):
            t = torch.tensor(temperatures,
                             dtype=logits.dtype,
                             device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities (before applying top-p and top-k).
        # no decoding stage so this is useless(maybe)
        logprobs = torch.log(probs)
        #max_prob_index = torch.argmax(logprobs, dim = -1)
        #max_prob_index = torch.argmax(probs, dim = -1)
        # Apply top-p and top-k truncation.
        top_ps, top_ks = _get_top_p_top_k(sampling_params, self.vocab_size)
        
        do_top_p = any(top_p < 1.0 - _SAMPLING_EPS for top_p in top_ps)
        do_top_k = any(top_k != self.vocab_size for top_k in top_ks)
        if do_top_p or do_top_k:
            probs = _apply_top_p_top_k(probs, top_ps, top_ks)

        # Sample the next tokens.
        new_token_ids = _sample(probs, sampling_params)
        logprob = logprobs[:, new_token_ids].item()
        return (new_token_ids, logprob)

def _get_temperature(sampling_params: List[ChunkSamplingParams]) -> List[float]:
    # Collect the temperatures for the logits.
    # NOTE: Zero temperature means deterministic sampling
    # (i.e., greedy sampling or beam search).
    # Set the temperature to 1 to avoid division by zero.
    ans: List[float] = []
    for sampling_param in sampling_params:
        temperature = sampling_param.temperature
        if temperature < _SAMPLING_EPS:
            #just don't divide by 0
            temperature = 1.0
        ans.append(temperature)
    return ans

def _get_top_p_top_k(
    sampling_params: List[ChunkSamplingParams],
    vocab_size: int,
) -> Tuple[List[float], List[int]]:
    top_ps: List[float] = []
    top_ks: List[int] = []
    for sampling_param in sampling_params:
        top_p = sampling_param.top_p
        top_k = min(sampling_param.top_k, vocab_size)
        top_k = vocab_size if top_k == -1 else top_k
        top_ps.append(top_p)
        top_ks.append(top_k)
    return (top_ps, top_ks)

def _apply_top_p_top_k(
    probs: torch.Tensor,
    top_ps: List[float],
    top_ks: List[int],
) -> torch.Tensor:
    p = torch.tensor(top_ps, dtype=probs.dtype, device=probs.device)
    k = torch.tensor(top_ks, dtype=torch.int, device=probs.device)
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

    # Apply top-p.
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    top_p_mask = (probs_sum - probs_sort) > p.unsqueeze(dim=1)
    probs_sort[top_p_mask] = 0.0

    # Apply top-k.
    # Create a mask for the top-k elements.
    top_k_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
    top_k_mask = top_k_mask.expand(probs_idx.shape[0], -1)
    top_k_mask = top_k_mask >= k.unsqueeze(dim=1)
    probs_sort[top_k_mask] = 0.0

    # Re-sort the probabilities.
    probs = torch.gather(probs_sort,
                         dim=-1,
                         index=torch.argsort(probs_idx, dim=-1))
    return probs

def _sample_from_prompt(
    prob: torch.Tensor,
    chunksamlingparams: ChunkSamplingParams
) -> List[int]:
    if chunksamlingparams.use_beam_search:
        # Beam search.
        beam_width = chunksamlingparams.best_of
        _, next_token_ids = torch.topk(prob, beam_width)
        next_token_ids = next_token_ids.tolist()
    elif chunksamlingparams.temperature < _SAMPLING_EPS:
        # Greedy sampling.
        assert chunksamlingparams.best_of == 1
        next_token_id = torch.argmax(prob)
        next_token_ids = [next_token_id.item()]
    else:
        # Random sampling.
        # Sample `best_of` tokens for the prompt.
        num_seqs = chunksamlingparams.best_of
        next_token_ids = torch.multinomial(prob,
                                           num_samples=num_seqs,
                                           replacement=True)
        next_token_ids = next_token_ids.tolist()
    return next_token_ids

def _sample(
    probs: torch.Tensor,
    sampling_params: List[ChunkSamplingParams]
) -> List[int]:
    #prob = probs[0]
    #prob should be [m,n] even if m==1
    length = probs.shape[0]
    ans: List[int] = []
    for i in range(0, length):
        prob = probs[i]
        sampling_param = sampling_params[i]
        next_token_ids = _sample_from_prompt(prob, sampling_param)
        ans.extend(next_token_ids)      
    return ans