#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include "trans_config.h"

namespace py = pybind11;

PYBIND11_MODULE(trans, m) {
    py::class_<TransConfig>(m, "TransConfig")
        .def(py::init<int, int, torch::Dtype, int>(),
             py::arg("head_size"), py::arg("num_heads"), py::arg("dtype"), py::arg("cache_size_per_block"))
        .def_readwrite("head_size", &TransConfig::head_size)
        .def_readwrite("num_heads", &TransConfig::num_heads)
        .def_readwrite("dtype", &TransConfig::dtype)
        .def_readwrite("cache_size_per_block", &TransConfig::cache_size_per_block);


    py::class_<TransEngine>(m, "TransEngine")
        .def(py::init<const TransConfig&, const std::vector<torch::Tensor>&>(),
             py::arg("trans_config"), py::arg("gpu_cache"))
        .def("send_blocks", &TransEngine::send_blocks)
        .def("recv_blocks", &TransEngine::recv_blocks)
        .def("check_send_finished_events", &TransEngine::check_send_finished_events)
        .def("check_recv_finished_events", &TransEngine::check_recv_finished_events);


    py::enum_<TaskType>(m, "TaskType")
        .value("TRANSFER_SEND", TaskType::TRANSFER_SEND)
        .value("TRANSFER_RECV_BLOCKS", TaskType::TRANSFER_RECV_BLOCKS)
        .value("TRANSFER_CHECK_FINISHED", TaskType::TRANSFER_CHECK_FINISHED)
        .value("TRANSFER_CHECK_SEND_FINISHED", TaskType::TRANSFER_CHECK_SEND_FINISHED)
        .value("TRANSFER_CHECK_RECV_FINISHED", TaskType::TRANSFER_CHECK_RECV_FINISHED)
        .export_values();

    py::class_<TransferTaskMeta>(m, "TransferTaskMeta")
        .def(py::init<const std::string&, const std::string& >())
        .def_readwrite("channel", &TransferTaskMeta::channel)
        .def_readwrite("request_id", &TransferTaskMeta::request_id);


    py::class_<TransferTask>(m, "TransferTask")
        .def(py::init<const TransferTaskMeta&, const std::vector<int>&, const std::vector<int>&>())
        .def_readwrite("meta", &TransferTask::meta)
        .def_readwrite("blocks", &TransferTask::blocks)
        .def_readwrite("opposite_ranks", &TransferTask::opposite_ranks);

    py::class_<TransWorker>(m, "TransWorker")
        .def(py::init<const TransConfig&, const std::vector<torch::Tensor>&, int, int, int>(),
             py::arg("trans_config"), py::arg("gpu_cache"), py::arg("rank"), py::arg("local_rank"), py::arg("nccl_local_rank"))
        .def("add_tasks", &TransWorker::add_tasks)
        .def("get_transfer_results", &TransWorker::get_transfer_results);
}
