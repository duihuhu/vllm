#include <iostream>
#include <torch/extension.h>
#include <map>

void print_blocks(torch::Tensor& src,
  torch::Tensor& dst,
  const std::map<int64_t, int64_t>& block_mapping){
  // torch::Device src_device = src.device();
  // torch::Device dst_device = dst.device();
  // cudaMemcpyKind memcpy_type;
  // if (src_device.is_cuda() && dst_device.is_cuda()) {
  //   TORCH_CHECK(
  //     src_device.index() == dst_device.index(),
  //     "src and dst must be on the same GPU");
  //   memcpy_type = cudaMemcpyDeviceToDevice;
  // } else if (src_device.is_cuda() && dst_device.is_cpu()) {
  //   memcpy_type = cudaMemcpyDeviceToHost;
  // } else if (src_device.is_cpu() && dst_device.is_cuda()) {
  //   memcpy_type = cudaMemcpyHostToDevice;
  // } else {
  //   TORCH_CHECK(false, "Invalid device combination");
  // }

  void *src_ptr = src.data_ptr();
  void *dst_ptr = dst.data_ptr();

  float *l_dst_ptr = (float*)dst_ptr;
  const int64_t block_size_in_bytes = src.element_size() * src[0].numel();
  // printf("block size in bytes: %lld\n", block_size_in_bytes);
  // const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  for (const auto& pair : block_mapping) {
    int64_t src_block_number = pair.first;
    int64_t dst_block_number = pair.second;
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    printf("src address %p , src_block_number %lld , src_offset %lld , dst_prt %p , dst_block_number %lld , dst_offset %lld\n", \
      src_ptr, src_block_number, src_offset, dst_ptr, dst_block_number, dst_offset);
   std::cout<<dst[dst_block_number]<<std::endl;
    // for (int i = 0; i < block_size_in_bytes; ++i) {
    // printf("%f", *(l_dst_ptr + dst_offset + 0));
    // }

  //   cudaMemcpyAsync(
  //     dst_ptr + dst_offset,
  //     src_ptr + src_offset,
  //     block_size_in_bytes,
  //     memcpy_type,
  //     stream);
  }

};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("print_blocks", &print_blocks, "add two number");
}