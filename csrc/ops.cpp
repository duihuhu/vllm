#include <iostream>
#include <torch/extension.h>

void print_blocks(torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping){
  void *src_ptr = src.data_ptr();
  printf("%p\n", src_ptr);

};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("print_blocks", &print_blocks, "add two number");
}