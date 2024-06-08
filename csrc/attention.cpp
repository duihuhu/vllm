#include <torch/extension.h>
#include <c10/util/Optional.h>
#include <vector>

void single_query_cached_kv_attention(
  torch::Tensor& out,
  torch::Tensor& query,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  float scale,
  torch::Tensor& block_tables,
  torch::Tensor& context_lens,
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes);

void new_single_query_cached_kv_attention(
  torch::Tensor& out,             // [num_seqs, num_heads, head_size]
  torch::Tensor& query,           // [num_seqs, num_heads, head_size]

  std::vector<torch::Tensor>& key_caches, // [num_layers, num_heads, head_size/x, block_size, x]
  std::vector<torch::Tensor>& value_caches, // [num_layers, num_heads, head_size, block_size]

  float scale,
  torch::Tensor& block_tables,    // [num_seqs, max_num_blocks_per_seq]
  torch::Tensor& context_lens,    // [num_seqs]
  int block_size,
  int max_context_len,
  const c10::optional<torch::Tensor>& alibi_slopes,
  
  int layer_num
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "single_query_cached_kv_attention",
    &single_query_cached_kv_attention,
    "Compute the attention between an input query and the cached key/value tensors");
  m.def(
    "new_single_query_cached_kv_attention",
    &new_single_query_cached_kv_attention,
    "Compute the attention between an input query and the cached key/value tensors in agg-blocks");
}
