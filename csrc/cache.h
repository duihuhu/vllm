#pragma once

#include <torch/extension.h>

#include <map>
#include <vector>

void swap_agg_block(
  torch::Tensor& src,
  torch::Tensor& dst,
  const int64_t block_size_in_bytes);

void copy_blocks_agg(
  torch::Tensor& key_caches_addresses, 
  torch::Tensor& value_caches_addresses,
  torch::Tensor& data_type_tensor,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping,
  const int num_layers,
  const int numel_per_layer);

void reshape_and_cache_agg(
  torch::Tensor& key,           
  torch::Tensor& value,         
  torch::Tensor& key_cache_addresses,     
  torch::Tensor& value_cache_addresses,   
  torch::Tensor& slot_mapping,  
  const std::string& kv_cache_dtype,
  const int block_size,
  const int x,
  const int num_layer);

void swap_blocks(
  torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping);

void copy_blocks(
  std::vector<torch::Tensor>& key_caches,
  std::vector<torch::Tensor>& value_caches,
  const std::map<int64_t, std::vector<int64_t>>& block_mapping);

void reshape_and_cache(
  torch::Tensor& key,
  torch::Tensor& value,
  torch::Tensor& key_cache,
  torch::Tensor& value_cache,
  torch::Tensor& slot_mapping,
  const std::string& kv_cache_dtype);

// Just for unittest
void convert_fp8_e5m2(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache);

void convert_fp8_e5m2_agg(
  torch::Tensor& src_cache,
  torch::Tensor& dst_cache);
