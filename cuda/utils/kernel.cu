#pragma once

__device__ __host__ __forceinline__ float compute_scaling_factor(float opacity, float alpha_threshold, float exp_power) {
    // * Computes the scaling factor for 3DGRT's adaptive clamping
    if (opacity <= alpha_threshold) {
        return 0.0f;
    }

    float log_ratio = logf(opacity / alpha_threshold);
    if (exp_power == 1.0f) {
        return sqrtf(2.0f * log_ratio);
    }
    if (exp_power == 2.0f) {
        return sqrtf(sqrtf(4.0f * log_ratio));
    }
    if (exp_power == 3.0f) {
        return cbrtf(sqrtf(6.0f * log_ratio));
    }

    float k = 2.0f * exp_power;
    return powf(k * log_ratio, 1.0f / k);
}

__device__ __forceinline__ float eval_gaussian(float3 local_hit, float exp_power) {
    float d = dot(local_hit, local_hit);
    if (exp_power == 1.0f) {
        return expf(-0.5f * d);
    }
    if (exp_power == 2.0f) {
        return expf(-0.25f * d * d);
    }
    if (exp_power == 3.0f) {
        return expf(-(d * d * d) / 6.0f);
    }

    float k = 2.0f * exp_power;
    return expf(-powf(d, exp_power) / k);
}

__device__ __forceinline__ float compute_alpha(float guassval, float opacity, float alpha_threshold) {
    return MAX_ALPHA * guassval * opacity;
}
