#pragma once

#include "../utils/vec_math.h"

struct Config {
    const float *exp_power;
    const float *alpha_threshold;
    const float *t_threshold;
    const bool *jitter_primary_rays;
    const float *global_scale_factor;
    const float *eps_forward_normalization;
    const float *eps_scale_grad;
    const floatK *background_channels;
    const bool *render_depth;
    const bool *render_ellipsoids;
    const float *ellipsoid_min_opacity;
    const bool *rays_from_python;
    const bool *zero_grads;
    const bool *enable_sh;
    const bool *needs_ray_output;
    const bool *update_channels;
};

#ifndef __CUDACC__
#include "headers.h"

struct ConfigDataHolder : torch::CustomClassHolder {
    Tensor exp_power = torch::tensor({2}, CUDA_FLOAT32);
    Tensor alpha_threshold = torch::tensor({0.01}, CUDA_FLOAT32);
    Tensor t_threshold = torch::tensor({0.01}, CUDA_FLOAT32);
    Tensor jitter_primary_rays = torch::tensor({false}, CUDA_BOOL);
    Tensor global_scale_factor = torch::ones({1}, CUDA_FLOAT32);
    Tensor eps_forward_normalization = torch::tensor({1e-12}, CUDA_FLOAT32);
    Tensor eps_scale_grad = torch::tensor({1e-12f}, CUDA_FLOAT32);
    Tensor background_channels = torch::zeros({CHANNELS}, CUDA_FLOAT32);
    Tensor render_depth = torch::tensor({false}, CUDA_BOOL);
    Tensor render_ellipsoids = torch::tensor({false}, CUDA_BOOL);
    Tensor ellipsoid_min_opacity = torch::tensor({0.0f}, CUDA_FLOAT32);
    Tensor rays_from_python = torch::tensor({false}, CUDA_BOOL);
    Tensor zero_grads = torch::tensor({true}, CUDA_BOOL);
    Tensor enable_sh = torch::tensor({true}, CUDA_BOOL);
    Tensor has_pre_mlp = torch::tensor({false}, CUDA_BOOL);
    Tensor needs_ray_output = torch::tensor({false}, CUDA_BOOL);
    Tensor update_channels = torch::tensor({true}, CUDA_BOOL);

    Config reify() {
        return Config{
            .exp_power = reinterpret_cast<float *>(exp_power.data_ptr()),
            .alpha_threshold = reinterpret_cast<float *>(alpha_threshold.data_ptr()),
            .t_threshold = reinterpret_cast<float *>(t_threshold.data_ptr()),
            .jitter_primary_rays = reinterpret_cast<bool *>(jitter_primary_rays.data_ptr()),
            .global_scale_factor = reinterpret_cast<float *>(global_scale_factor.data_ptr()),
            .eps_forward_normalization = reinterpret_cast<float *>(eps_forward_normalization.data_ptr()),
            .eps_scale_grad = reinterpret_cast<float *>(eps_scale_grad.data_ptr()),
            .background_channels = reinterpret_cast<floatK *>(background_channels.data_ptr()),
            .render_depth = reinterpret_cast<bool *>(render_depth.data_ptr()),
            .render_ellipsoids = reinterpret_cast<bool *>(render_ellipsoids.data_ptr()),
            .ellipsoid_min_opacity = reinterpret_cast<float *>(ellipsoid_min_opacity.data_ptr()),
            .rays_from_python = reinterpret_cast<bool *>(rays_from_python.data_ptr()),
            .zero_grads = reinterpret_cast<bool *>(zero_grads.data_ptr()),
            .enable_sh = reinterpret_cast<bool *>(enable_sh.data_ptr()),
            .needs_ray_output = reinterpret_cast<bool *>(needs_ray_output.data_ptr()),
            .update_channels = reinterpret_cast<bool *>(update_channels.data_ptr()),
        };
    }

    static void bind(torch::Library &m) {
        m.class_<ConfigDataHolder>("ConfigDataHolder")
            .def_readonly("exp_power", &ConfigDataHolder::exp_power)
            .def_readonly("alpha_threshold", &ConfigDataHolder::alpha_threshold)
            .def_readonly("t_threshold", &ConfigDataHolder::t_threshold)
            .def_readonly("jitter_primary_rays", &ConfigDataHolder::jitter_primary_rays)
            .def_readonly("global_scale_factor", &ConfigDataHolder::global_scale_factor)
            .def_readonly("eps_forward_normalization", &ConfigDataHolder::eps_forward_normalization)
            .def_readonly("eps_scale_grad", &ConfigDataHolder::eps_scale_grad)
            .def_readonly("background_channels", &ConfigDataHolder::background_channels)
            .def_readonly("render_depth", &ConfigDataHolder::render_depth)
            .def_readonly("render_ellipsoids", &ConfigDataHolder::render_ellipsoids)
            .def_readonly("ellipsoid_min_opacity", &ConfigDataHolder::ellipsoid_min_opacity)
            .def_readonly("rays_from_python", &ConfigDataHolder::rays_from_python)
            .def_readonly("zero_grads", &ConfigDataHolder::zero_grads)
            .def_readonly("enable_sh", &ConfigDataHolder::enable_sh)
            .def_readonly("needs_ray_output", &ConfigDataHolder::needs_ray_output)
            .def_readonly("update_channels", &ConfigDataHolder::update_channels);
    }
};
#endif
