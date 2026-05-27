#pragma once

#include <cuda_fp16.h>

#include "../utils/vec_math.h"

struct Gaussians {
    int count;
    int max_sh_degree;
    int num_sh_coeffs;
    int *current_sh_degree;

    float3 *__restrict__ mean;
    float4 *__restrict__ rotation;
    float3 *__restrict__ scale;
    float *__restrict__ opacity;
    floatK *__restrict__ channels;
    float3 *__restrict__ sh_coeffs_dc;
    float3 *__restrict__ sh_coeffs_rest;

    float *__restrict__ lr_mean;
    float *__restrict__ lr_rotation;
    float *__restrict__ lr_scale;
    float *__restrict__ lr_opacity;
    float *__restrict__ lr_channels;
    float *__restrict__ lr_sh_dc;
    float *__restrict__ lr_sh_rest;

    float *__restrict__ beta_1;
    float *__restrict__ beta_2;
    float *__restrict__ epsilon;
    int *__restrict__ sh_update_laziness;

    float3 *__restrict__ grad_mean;
    float4 *__restrict__ grad_rotation;
    float3 *__restrict__ grad_scale;
    float *__restrict__ grad_opacity;
    floatK *__restrict__ grad_channels;
    float3 *__restrict__ grad_sh_coeffs_dc;
    float3 *__restrict__ grad_sh_coeffs_rest;

    float3 *__restrict__ first_moment_mean;
    float4 *__restrict__ first_moment_rotation;
    float3 *__restrict__ first_moment_scale;
    float *__restrict__ first_moment_opacity;
    floatK *__restrict__ first_moment_channels;
    float3 *__restrict__ first_moment_sh_coeffs_dc;
    float3 *__restrict__ first_moment_sh_coeffs_rest;

    float3 *__restrict__ second_moment_mean;
    float4 *__restrict__ second_moment_rotation;
    float3 *__restrict__ second_moment_scale;
    float *__restrict__ second_moment_opacity;
    floatK *__restrict__ second_moment_channels;
    float3 *__restrict__ second_moment_sh_coeffs_dc;
    float3 *__restrict__ second_moment_sh_coeffs_rest;

    float *__restrict__ pruning_weight;
    int *__restrict__ pruning_counter;

    bool *__restrict__ was_visible;
    float *__restrict__ marked;

    bool sh_is_fp16;
};

#ifndef __CUDACC__
#include "headers.h"

struct GaussianDataHolder : torch::CustomClassHolder {
    int count = 1;
    int max_sh_degree;
    int num_sh_coeffs;
    bool inference_only;
    Tensor current_sh_degree = torch::tensor({0}, CUDA_INT32);

    Tensor mean = torch::zeros({1, 3}, CUDA_FLOAT32);
    Tensor rotation = torch::zeros({1, 4}, CUDA_FLOAT32);
    Tensor scale = torch::zeros({1, 3}, CUDA_FLOAT32);
    Tensor opacity = torch::zeros({1, 1}, CUDA_FLOAT32);
    Tensor channels = torch::zeros({1, CHANNELS}, CUDA_FLOAT32);
    Tensor sh_coeffs_dc;
    Tensor sh_coeffs_rest;

    Tensor lr_mean = torch::ones({1}, CUDA_FLOAT32);
    Tensor lr_rotation = torch::ones({1}, CUDA_FLOAT32);
    Tensor lr_scale = torch::ones({1}, CUDA_FLOAT32);
    Tensor lr_opacity = torch::ones({1}, CUDA_FLOAT32);
    Tensor lr_channels = torch::ones({1}, CUDA_FLOAT32);
    Tensor lr_sh_dc = torch::ones({1}, CUDA_FLOAT32);
    Tensor lr_sh_rest = torch::ones({1}, CUDA_FLOAT32);

    Tensor beta_1 = torch::tensor({0.9}, CUDA_FLOAT32);
    Tensor beta_2 = torch::tensor({0.999}, CUDA_FLOAT32);
    Tensor epsilon = torch::tensor({1e-15}, CUDA_FLOAT32);
    Tensor sh_update_laziness = torch::tensor({1}, CUDA_INT32);

    Tensor grad_mean;
    Tensor grad_rotation;
    Tensor grad_scale;
    Tensor grad_opacity;
    Tensor grad_channels;
    Tensor grad_sh_coeffs_dc;
    Tensor grad_sh_coeffs_rest;

    Tensor first_moment_mean;
    Tensor first_moment_rotation;
    Tensor first_moment_scale;
    Tensor first_moment_opacity;
    Tensor first_moment_channels;
    Tensor first_moment_sh_coeffs_dc;
    Tensor first_moment_sh_coeffs_rest;

    Tensor second_moment_mean;
    Tensor second_moment_rotation;
    Tensor second_moment_scale;
    Tensor second_moment_opacity;
    Tensor second_moment_channels;
    Tensor second_moment_sh_coeffs_dc;
    Tensor second_moment_sh_coeffs_rest;

    Tensor pruning_weight;
    Tensor pruning_counter;

    Tensor was_visible;
    Tensor marked;

    GaussianDataHolder(int64_t max_sh_degree, bool inference_only_ = false)
        : max_sh_degree(max_sh_degree), num_sh_coeffs((max_sh_degree + 1) * (max_sh_degree + 1) - 1),
          inference_only(inference_only_) {
        int num_sh_coeffs = (max_sh_degree + 1) * (max_sh_degree + 1) - 1;
        auto CUDA_SH_DTYPE = inference_only ? CUDA_FLOAT16 : CUDA_FLOAT32;
        sh_coeffs_dc = torch::zeros({1, 1, CHANNELS}, CUDA_SH_DTYPE);
        sh_coeffs_rest = torch::zeros({1, num_sh_coeffs, CHANNELS}, CUDA_SH_DTYPE);
        if (!inference_only) {
            grad_mean = torch::zeros({1, 3}, CUDA_FLOAT32);
            grad_rotation = torch::zeros({1, 4}, CUDA_FLOAT32);
            grad_scale = torch::zeros({1, 3}, CUDA_FLOAT32);
            grad_opacity = torch::zeros({1, 1}, CUDA_FLOAT32);
            grad_channels = torch::zeros({1, CHANNELS}, CUDA_FLOAT32);
            grad_sh_coeffs_dc = torch::zeros({1, 1, CHANNELS}, CUDA_FLOAT32);
            grad_sh_coeffs_rest = torch::zeros({1, num_sh_coeffs, CHANNELS}, CUDA_FLOAT32);
            first_moment_mean = torch::zeros({1, 3}, CUDA_FLOAT32);
            first_moment_rotation = torch::zeros({1, 4}, CUDA_FLOAT32);
            first_moment_scale = torch::zeros({1, 3}, CUDA_FLOAT32);
            first_moment_opacity = torch::zeros({1, 1}, CUDA_FLOAT32);
            first_moment_channels = torch::zeros({1, CHANNELS}, CUDA_FLOAT32);
            first_moment_sh_coeffs_dc = torch::zeros({1, 1, CHANNELS}, CUDA_FLOAT32);
            first_moment_sh_coeffs_rest = torch::zeros({1, num_sh_coeffs, CHANNELS}, CUDA_FLOAT32);
            second_moment_mean = torch::zeros({1, 3}, CUDA_FLOAT32);
            second_moment_rotation = torch::zeros({1, 4}, CUDA_FLOAT32);
            second_moment_scale = torch::zeros({1, 3}, CUDA_FLOAT32);
            second_moment_opacity = torch::zeros({1, 1}, CUDA_FLOAT32);
            second_moment_channels = torch::zeros({1, CHANNELS}, CUDA_FLOAT32);
            second_moment_sh_coeffs_dc = torch::zeros({1, 1, CHANNELS}, CUDA_FLOAT32);
            second_moment_sh_coeffs_rest = torch::zeros({1, num_sh_coeffs, CHANNELS}, CUDA_FLOAT32);
            pruning_weight = torch::zeros({1, 1}, CUDA_FLOAT32);
            pruning_counter = torch::zeros({1, 1}, CUDA_INT32);
            was_visible = torch::zeros({1, 1}, CUDA_BOOL);
            marked = torch::zeros({1, 1}, CUDA_FLOAT32);

            mean.mutable_grad() = grad_mean;
            rotation.mutable_grad() = grad_rotation;
            scale.mutable_grad() = grad_scale;
            opacity.mutable_grad() = grad_opacity;
            channels.mutable_grad() = grad_channels;
            sh_coeffs_dc.mutable_grad() = grad_sh_coeffs_dc;
            sh_coeffs_rest.mutable_grad() = grad_sh_coeffs_rest;
        } else {
            grad_mean = torch::zeros({1, 3}, CUDA_FLOAT32);
            grad_rotation = torch::zeros({1, 4}, CUDA_FLOAT32);
            grad_scale = torch::zeros({1, 3}, CUDA_FLOAT32);
            grad_opacity = torch::zeros({1, 1}, CUDA_FLOAT32);
            grad_channels = torch::zeros({1, CHANNELS}, CUDA_FLOAT32);
            grad_sh_coeffs_dc = torch::zeros({1, 1, CHANNELS}, CUDA_FLOAT32);
            grad_sh_coeffs_rest = torch::zeros({1, num_sh_coeffs, CHANNELS}, CUDA_FLOAT32);
            first_moment_mean = torch::zeros({1, 3}, CUDA_FLOAT32);
            first_moment_rotation = torch::zeros({1, 4}, CUDA_FLOAT32);
            first_moment_scale = torch::zeros({1, 3}, CUDA_FLOAT32);
            first_moment_opacity = torch::zeros({1, 1}, CUDA_FLOAT32);
            first_moment_channels = torch::zeros({1, CHANNELS}, CUDA_FLOAT32);
            first_moment_sh_coeffs_dc = torch::zeros({1, 1, CHANNELS}, CUDA_FLOAT32);
            first_moment_sh_coeffs_rest = torch::zeros({1, num_sh_coeffs, CHANNELS}, CUDA_FLOAT32);
            second_moment_mean = torch::zeros({1, 3}, CUDA_FLOAT32);
            second_moment_rotation = torch::zeros({1, 4}, CUDA_FLOAT32);
            second_moment_scale = torch::zeros({1, 3}, CUDA_FLOAT32);
            second_moment_opacity = torch::zeros({1, 1}, CUDA_FLOAT32);
            second_moment_channels = torch::zeros({1, CHANNELS}, CUDA_FLOAT32);
            second_moment_sh_coeffs_dc = torch::zeros({1, 1, CHANNELS}, CUDA_FLOAT32);
            second_moment_sh_coeffs_rest = torch::zeros({1, num_sh_coeffs, CHANNELS}, CUDA_FLOAT32);
            pruning_weight = torch::zeros({1, 1}, CUDA_FLOAT32);
            pruning_counter = torch::zeros({1, 1}, CUDA_INT32);
            was_visible = torch::zeros({1, 1}, CUDA_BOOL);
            marked = torch::zeros({1, 1}, CUDA_FLOAT32);
        }
    }

    void resize(int64_t num_new_gaussians) {
        if (count > 1) {
            TORCH_CHECK(num_new_gaussians <= count,
                        "The new number of Gaussians must not be larger than the current count.");
        }

        torch::NoGradGuard no_grad;

        count = num_new_gaussians;

        mean.resize_({num_new_gaussians, 3});
        rotation.resize_({num_new_gaussians, 4});
        scale.resize_({num_new_gaussians, 3});
        opacity.resize_({num_new_gaussians, 1});
        channels.resize_({num_new_gaussians, CHANNELS});
        sh_coeffs_dc.resize_({num_new_gaussians, 1, CHANNELS});
        sh_coeffs_rest.resize_({num_new_gaussians, num_sh_coeffs, CHANNELS});

        lr_mean.resize_({num_new_gaussians, 1});
        lr_rotation.resize_({num_new_gaussians, 1});
        lr_scale.resize_({num_new_gaussians, 1});
        lr_opacity.resize_({num_new_gaussians, 1});
        lr_channels.resize_({num_new_gaussians, 1});
        lr_sh_dc.resize_({num_new_gaussians, 1});
        lr_sh_rest.resize_({num_new_gaussians, 1});

        if (!inference_only) {
            grad_mean.resize_({num_new_gaussians, 3});
            grad_rotation.resize_({num_new_gaussians, 4});
            grad_scale.resize_({num_new_gaussians, 3});
            grad_opacity.resize_({num_new_gaussians, 1});
            grad_channels.resize_({num_new_gaussians, CHANNELS});
            grad_sh_coeffs_dc.resize_({num_new_gaussians, 1, CHANNELS});
            grad_sh_coeffs_rest.resize_({num_new_gaussians, num_sh_coeffs, CHANNELS});

            first_moment_mean.resize_({num_new_gaussians, 3});
            first_moment_rotation.resize_({num_new_gaussians, 4});
            first_moment_scale.resize_({num_new_gaussians, 3});
            first_moment_opacity.resize_({num_new_gaussians, 1});
            first_moment_channels.resize_({num_new_gaussians, CHANNELS});
            first_moment_sh_coeffs_dc.resize_({num_new_gaussians, 1, CHANNELS});
            first_moment_sh_coeffs_rest.resize_({num_new_gaussians, num_sh_coeffs, CHANNELS});

            second_moment_mean.resize_({num_new_gaussians, 3});
            second_moment_rotation.resize_({num_new_gaussians, 4});
            second_moment_scale.resize_({num_new_gaussians, 3});
            second_moment_opacity.resize_({num_new_gaussians, 1});
            second_moment_channels.resize_({num_new_gaussians, CHANNELS});
            second_moment_sh_coeffs_dc.resize_({num_new_gaussians, 1, CHANNELS});
            second_moment_sh_coeffs_rest.resize_({num_new_gaussians, num_sh_coeffs, CHANNELS});

            pruning_weight.resize_({num_new_gaussians, 1});
            pruning_counter.resize_({num_new_gaussians, 1});

            was_visible.resize_({num_new_gaussians, 1});
            marked.resize_({num_new_gaussians, 1});

            mean.mutable_grad() = grad_mean;
            rotation.mutable_grad() = grad_rotation;
            scale.mutable_grad() = grad_scale;
            opacity.mutable_grad() = grad_opacity;
            channels.mutable_grad() = grad_channels;
            sh_coeffs_dc.mutable_grad() = grad_sh_coeffs_dc;
            sh_coeffs_rest.mutable_grad() = grad_sh_coeffs_rest;
        }
    }

    Gaussians reify() {
        Gaussians gaussians;

        gaussians.count = count;
        gaussians.max_sh_degree = max_sh_degree;
        gaussians.num_sh_coeffs = num_sh_coeffs;
        gaussians.current_sh_degree = reinterpret_cast<int *>(current_sh_degree.data_ptr());

        gaussians.mean = reinterpret_cast<float3 *>(mean.data_ptr());
        gaussians.rotation = reinterpret_cast<float4 *>(rotation.data_ptr());
        gaussians.scale = reinterpret_cast<float3 *>(scale.data_ptr());
        gaussians.opacity = reinterpret_cast<float *>(opacity.data_ptr());
        gaussians.channels = reinterpret_cast<floatK *>(channels.data_ptr());
        gaussians.sh_coeffs_dc = reinterpret_cast<float3 *>(sh_coeffs_dc.data_ptr());
        gaussians.sh_coeffs_rest = reinterpret_cast<float3 *>(sh_coeffs_rest.data_ptr());

        gaussians.lr_mean = reinterpret_cast<float *>(lr_mean.data_ptr());
        gaussians.lr_rotation = reinterpret_cast<float *>(lr_rotation.data_ptr());
        gaussians.lr_scale = reinterpret_cast<float *>(lr_scale.data_ptr());
        gaussians.lr_opacity = reinterpret_cast<float *>(lr_opacity.data_ptr());
        gaussians.lr_channels = reinterpret_cast<float *>(lr_channels.data_ptr());
        gaussians.lr_sh_dc = reinterpret_cast<float *>(lr_sh_dc.data_ptr());
        gaussians.lr_sh_rest = reinterpret_cast<float *>(lr_sh_rest.data_ptr());

        gaussians.beta_1 = reinterpret_cast<float *>(beta_1.data_ptr());
        gaussians.beta_2 = reinterpret_cast<float *>(beta_2.data_ptr());
        gaussians.epsilon = reinterpret_cast<float *>(epsilon.data_ptr());
        gaussians.sh_update_laziness = reinterpret_cast<int *>(sh_update_laziness.data_ptr());

        gaussians.grad_mean = !inference_only ? reinterpret_cast<float3 *>(grad_mean.data_ptr()) : nullptr;
        gaussians.grad_rotation =
            !inference_only ? reinterpret_cast<float4 *>(grad_rotation.data_ptr()) : nullptr;
        gaussians.grad_scale =
            !inference_only ? reinterpret_cast<float3 *>(grad_scale.data_ptr()) : nullptr;
        gaussians.grad_opacity =
            !inference_only ? reinterpret_cast<float *>(grad_opacity.data_ptr()) : nullptr;
        gaussians.grad_channels =
            !inference_only ? reinterpret_cast<floatK *>(grad_channels.data_ptr()) : nullptr;
        gaussians.grad_sh_coeffs_dc =
            !inference_only ? reinterpret_cast<float3 *>(grad_sh_coeffs_dc.data_ptr()) : nullptr;
        gaussians.grad_sh_coeffs_rest =
            !inference_only ? reinterpret_cast<float3 *>(grad_sh_coeffs_rest.data_ptr()) : nullptr;

        gaussians.first_moment_mean =
            !inference_only ? reinterpret_cast<float3 *>(first_moment_mean.data_ptr()) : nullptr;
        gaussians.first_moment_rotation =
            !inference_only ? reinterpret_cast<float4 *>(first_moment_rotation.data_ptr()) : nullptr;
        gaussians.first_moment_scale =
            !inference_only ? reinterpret_cast<float3 *>(first_moment_scale.data_ptr()) : nullptr;
        gaussians.first_moment_opacity =
            !inference_only ? reinterpret_cast<float *>(first_moment_opacity.data_ptr()) : nullptr;
        gaussians.first_moment_channels =
            !inference_only ? reinterpret_cast<floatK *>(first_moment_channels.data_ptr()) : nullptr;
        gaussians.first_moment_sh_coeffs_dc = !inference_only
                                                  ? reinterpret_cast<float3 *>(first_moment_sh_coeffs_dc.data_ptr())
                                                  : nullptr;
        gaussians.first_moment_sh_coeffs_rest = !inference_only
                                                    ? reinterpret_cast<float3 *>(first_moment_sh_coeffs_rest.data_ptr())
                                                    : nullptr;

        gaussians.second_moment_mean =
            !inference_only ? reinterpret_cast<float3 *>(second_moment_mean.data_ptr()) : nullptr;
        gaussians.second_moment_rotation =
            !inference_only ? reinterpret_cast<float4 *>(second_moment_rotation.data_ptr()) : nullptr;
        gaussians.second_moment_scale =
            !inference_only ? reinterpret_cast<float3 *>(second_moment_scale.data_ptr()) : nullptr;
        gaussians.second_moment_opacity =
            !inference_only ? reinterpret_cast<float *>(second_moment_opacity.data_ptr()) : nullptr;
        gaussians.second_moment_channels =
            !inference_only ? reinterpret_cast<floatK *>(second_moment_channels.data_ptr()) : nullptr;
        gaussians.second_moment_sh_coeffs_dc = !inference_only
                                                   ? reinterpret_cast<float3 *>(second_moment_sh_coeffs_dc.data_ptr())
                                                   : nullptr;
        gaussians.second_moment_sh_coeffs_rest = !inference_only
                                                     ? reinterpret_cast<float3 *>(second_moment_sh_coeffs_rest.data_ptr())
                                                     : nullptr;

        gaussians.pruning_weight =
            !inference_only ? reinterpret_cast<float *>(pruning_weight.data_ptr()) : nullptr;
        gaussians.pruning_counter =
            !inference_only ? reinterpret_cast<int *>(pruning_counter.data_ptr()) : nullptr;

        gaussians.was_visible =
            !inference_only ? reinterpret_cast<bool *>(was_visible.data_ptr()) : nullptr;
        gaussians.marked = !inference_only ? reinterpret_cast<float *>(marked.data_ptr()) : nullptr;

        gaussians.sh_is_fp16 = inference_only;

        return gaussians;
    }

    void increment_sh_degree() {
        torch::NoGradGuard no_grad;
        if (current_sh_degree.item<int64_t>() < max_sh_degree) {
            current_sh_degree += 1;
        }
    }

    static void bind(torch::Library &m) {
        m.class_<GaussianDataHolder>("GaussianDataHolder")
            .def("increment_sh_degree", &GaussianDataHolder::increment_sh_degree)
            .def_readonly("current_sh_degree", &GaussianDataHolder::current_sh_degree)

            .def_readonly("mean", &GaussianDataHolder::mean)
            .def_readonly("rotation", &GaussianDataHolder::rotation)
            .def_readonly("scale", &GaussianDataHolder::scale)
            .def_readonly("opacity", &GaussianDataHolder::opacity)
            .def_readonly("channels", &GaussianDataHolder::channels)
            .def_readonly("sh_coeffs_dc", &GaussianDataHolder::sh_coeffs_dc)
            .def_readonly("sh_coeffs_rest", &GaussianDataHolder::sh_coeffs_rest)

            .def_readonly("lr_mean", &GaussianDataHolder::lr_mean)
            .def_readonly("lr_rotation", &GaussianDataHolder::lr_rotation)
            .def_readonly("lr_scale", &GaussianDataHolder::lr_scale)
            .def_readonly("lr_opacity", &GaussianDataHolder::lr_opacity)
            .def_readonly("lr_channels", &GaussianDataHolder::lr_channels)
            .def_readonly("lr_sh_dc", &GaussianDataHolder::lr_sh_dc)
            .def_readonly("lr_sh_rest", &GaussianDataHolder::lr_sh_rest)

            .def_readonly("beta_1", &GaussianDataHolder::beta_1)
            .def_readonly("beta_2", &GaussianDataHolder::beta_2)
            .def_readonly("epsilon", &GaussianDataHolder::epsilon)
            .def_readonly("sh_update_laziness", &GaussianDataHolder::sh_update_laziness)

            .def_readonly("grad_mean", &GaussianDataHolder::grad_mean)
            .def_readonly("grad_rotation", &GaussianDataHolder::grad_rotation)
            .def_readonly("grad_scale", &GaussianDataHolder::grad_scale)
            .def_readonly("grad_opacity", &GaussianDataHolder::grad_opacity)
            .def_readonly("grad_channels", &GaussianDataHolder::grad_channels)
            .def_readonly("grad_sh_coeffs_dc", &GaussianDataHolder::grad_sh_coeffs_dc)
            .def_readonly("grad_sh_coeffs_rest", &GaussianDataHolder::grad_sh_coeffs_rest)

            .def_readonly("first_moment_mean", &GaussianDataHolder::first_moment_mean)
            .def_readonly("first_moment_rotation", &GaussianDataHolder::first_moment_rotation)
            .def_readonly("first_moment_scale", &GaussianDataHolder::first_moment_scale)
            .def_readonly("first_moment_opacity", &GaussianDataHolder::first_moment_opacity)
            .def_readonly("first_moment_channels", &GaussianDataHolder::first_moment_channels)
            .def_readonly("first_moment_sh_coeffs_dc", &GaussianDataHolder::first_moment_sh_coeffs_dc)
            .def_readonly("first_moment_sh_coeffs_rest", &GaussianDataHolder::first_moment_sh_coeffs_rest)

            .def_readonly("second_moment_mean", &GaussianDataHolder::second_moment_mean)
            .def_readonly("second_moment_rotation", &GaussianDataHolder::second_moment_rotation)
            .def_readonly("second_moment_scale", &GaussianDataHolder::second_moment_scale)
            .def_readonly("second_moment_opacity", &GaussianDataHolder::second_moment_opacity)
            .def_readonly("second_moment_channels", &GaussianDataHolder::second_moment_channels)
            .def_readonly("second_moment_sh_coeffs_dc", &GaussianDataHolder::second_moment_sh_coeffs_dc)
            .def_readonly("second_moment_sh_coeffs_rest", &GaussianDataHolder::second_moment_sh_coeffs_rest)

            .def_readonly("pruning_weight", &GaussianDataHolder::pruning_weight)
            .def_readonly("pruning_counter", &GaussianDataHolder::pruning_counter)

            .def_readonly("was_visible", &GaussianDataHolder::was_visible)
            .def_readonly("marked", &GaussianDataHolder::marked);
    }
};
#endif
