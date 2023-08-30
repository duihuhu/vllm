#include <iostream>
#include <torch/extension.h>

void print_blocks(){
  printf("aaa\n");
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("print_blocks", &print_blocks, "add two number");
}