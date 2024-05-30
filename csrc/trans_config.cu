#include "trans_config.h"
TransConfig::TransConfig(){}
TransConfig::TransConfig(int head_size, int num_heads, torch::Dtype dtype, int cache_size_per_block)
    : head_size(head_size),
      num_heads(num_heads),
      dtype(dtype),
      cache_size_per_block(cache_size_per_block) {}

