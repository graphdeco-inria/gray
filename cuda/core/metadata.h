#pragma once

struct Metadata {
    bool *__restrict__ run_forward_pass;
    bool *__restrict__ run_backward_pass;
    bool *__restrict__ grads_enabled;

    uint32_t *__restrict__ total_num_calls;
    uint32_t *__restrict__ total_num_steps;
};

#ifndef __CUDACC__
#include "headers.h"

struct MetaDataHolder : torch::CustomClassHolder {
    Tensor run_forward_pass = torch::ones({1}, CUDA_BOOL);
    Tensor run_backward_pass = torch::ones({1}, CUDA_BOOL);
    Tensor grads_enabled = torch::ones({1}, CUDA_BOOL);
    Tensor total_num_calls = torch::zeros({1}, CUDA_INT32);
    Tensor total_num_steps = torch::zeros({1}, CUDA_INT32);

    MetaDataHolder(uint32_t image_width, uint32_t image_height) {}

    Metadata reify() {
        return Metadata{.run_forward_pass = reinterpret_cast<bool *>(run_forward_pass.data_ptr()),
                        .run_backward_pass = reinterpret_cast<bool *>(run_backward_pass.data_ptr()),
                        .grads_enabled = reinterpret_cast<bool *>(grads_enabled.data_ptr()),
                        .total_num_calls = reinterpret_cast<uint32_t *>(total_num_calls.data_ptr()),
                        .total_num_steps = reinterpret_cast<uint32_t *>(total_num_steps.data_ptr())};
    }

    void update() {
        bool grad_is_enabled = torch::autograd::GradMode::is_enabled();
        grads_enabled.fill_(grad_is_enabled);
        total_num_calls += 1;
        if (grad_is_enabled) {
            total_num_steps += 1;
        }
    }

    static void bind(torch::Library &m) {
        m.class_<MetaDataHolder>("MetaDataHolder")
            .def_readonly("grads_enabled", &MetaDataHolder::grads_enabled)
            .def_readonly("total_num_calls", &MetaDataHolder::total_num_calls)
            .def_readonly("total_num_steps", &MetaDataHolder::total_num_steps)
            .def("update", &MetaDataHolder::update);
    }
};

#endif