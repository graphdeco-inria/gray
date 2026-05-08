#pragma once

// * Helpers that auto-apply the correct activation for gaussian parameters. Kept in a separate file to avoid circular
// dependencies.

#include "activations.cu"
#include "../params.h"

__device__ __forceinline__ auto read_opacity(const Params &params, int gaussian_id) {
    return sigmoid_act(params.gaussians.opacity[gaussian_id]);
}
__device__ __forceinline__ auto read_scale(const Params &params, int gaussian_id) {
    return exp_act(params.gaussians.scale[gaussian_id]);
}
__device__ __forceinline__ auto read_mean(const Params &params, int gaussian_id) {
    return identity_act(params.gaussians.mean[gaussian_id]);
}
__device__ __forceinline__ auto read_rotation(const Params &params, int gaussian_id) {
    return normalize_act(params.gaussians.rotation[gaussian_id]);
}
__device__ __forceinline__ auto read_channels(const Params &params, int gaussian_id) {
    return params.gaussians.channels[gaussian_id];
}

__device__ __forceinline__ auto backward_act_for_opacity(auto grad_value, auto value) {
    return backward_sigmoid_act(grad_value, value);
}
__device__ __forceinline__ auto backward_act_for_scale(auto grad_value, auto value) {
    return backward_exp_act(grad_value, value);
}
__device__ __forceinline__ auto backward_act_for_mean(auto grad_value, auto value) {
    return backward_identity_act(grad_value, value);
}
__device__ __forceinline__ auto backward_act_for_channels(auto grad_value, auto value) { return grad_value; }
