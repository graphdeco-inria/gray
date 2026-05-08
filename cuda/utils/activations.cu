#pragma once

// * Sigmoid

__device__ float sigmoid_act(float x) { return 1.0f / (1.0f + exp(-x)); }
__device__ float3 sigmoid_act(float3 x) { return {sigmoid_act(x.x), sigmoid_act(x.y), sigmoid_act(x.z)}; }

__device__ float backward_sigmoid_act(float grad_y, float y) { return grad_y * y * (1.0f - y); }
__device__ float3 backward_sigmoid_act(float3 grad_y, float3 y) {
    return {backward_sigmoid_act(grad_y.x, y.x), backward_sigmoid_act(grad_y.y, y.y),
            backward_sigmoid_act(grad_y.z, y.z)};
}

// * Softplus

__device__ float softplus_act(float x) { return log(1.0f + exp(x)); }
__device__ float3 softplus_act(float3 x) { return {softplus_act(x.x), softplus_act(x.y), softplus_act(x.z)}; }

__device__ float backward_softplus_act(float grad_y, float x, float y) { return grad_y / (1.0f + exp(-x)); }
__device__ float3 backward_softplus_act(float3 grad_y, float3 x, float3 y) {
    return {backward_softplus_act(grad_y.x, x.x, y.x), backward_softplus_act(grad_y.y, x.y, y.y),
            backward_softplus_act(grad_y.z, x.z, y.z)};
}

// * ReLU

__device__ float relu_act(float x) { return max(0.0f, x); }

__device__ float backward_relu_act(float grad_y, float y) { return grad_y * (y >= 0.0f ? 1.0f : 0.0f); }

__device__ float3 relu_act(float3 x) { return {relu_act(x.x), relu_act(x.y), relu_act(x.z)}; }

__device__ float3 backward_relu_act(float3 grad_y, float3 y) {
    return {backward_relu_act(grad_y.x, y.x), backward_relu_act(grad_y.y, y.y), backward_relu_act(grad_y.z, y.z)};
}

// * Clipped ReLU

__device__ float clipped_relu_act(float x) { return min(max(0.0f, x), 1.0f); }

__device__ float backward_clipped_relu_act(float grad_y, float y) {
    return grad_y * (y >= 0.0f && y <= 1.0f ? 1.0f : 0.0f);
}

__device__ float3 clipped_relu_act(float3 x) {
    return {clipped_relu_act(x.x), clipped_relu_act(x.y), clipped_relu_act(x.z)};
}

__device__ float3 backward_clipped_relu_act(float3 grad_y, float3 y) {
    return {backward_clipped_relu_act(grad_y.x, y.x), backward_clipped_relu_act(grad_y.y, y.y),
            backward_clipped_relu_act(grad_y.z, y.z)};
}

// * Exp

__device__ float exp_act(float x) { return exp(x); }
__device__ float3 exp_act(float3 x) { return {exp_act(x.x), exp_act(x.y), exp_act(x.z)}; }

__device__ float backward_exp_act(float grad_y, float y) { return grad_y * y; }
__device__ float3 backward_exp_act(float3 grad_y, float3 y) {
    return {backward_exp_act(grad_y.x, y.x), backward_exp_act(grad_y.y, y.y), backward_exp_act(grad_y.z, y.z)};
}

__device__ float4 normalize_act(float4 x) {
    float norm = length(x);
    return {x.x / norm, x.y / norm, x.z / norm, x.w / norm};
}
__device__ float4 backward_normalize_act(float4 grad_y, float4 x, float4 y) {
    return dot(grad_y, x) * -x / powf(length(x), 3) + grad_y / length(x);
}

// * Identity

__device__ float identity_act(float x) { return x; }

__device__ float3 identity_act(float3 x) { return {x.x, x.y, x.z}; }

__device__ float backward_identity_act(float grad_y, float y) { return grad_y; }

__device__ float3 backward_identity_act(float3 grad_y, float3 y) { return {grad_y.x, grad_y.y, grad_y.z}; }
