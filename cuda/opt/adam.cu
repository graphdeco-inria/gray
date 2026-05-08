
#include "../core/gaussians.h"
#include "../utils/vec_math.h"
#include <cstdio>

struct Adam {
    int step;
    bool zero_grads;
    float beta1;
    float beta2;
    float epsilon;

    template <typename T> __device__ void operator()(T &p, T &m, T &v, T &g, float lr) {
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * g * g;
        auto m_hat = m / (1.0f - powf(beta1, step + 1.0f));
        auto v_hat = v / (1.0f - powf(beta2, step + 1.0f));
        p -= lr * m_hat / (sqrtf(v_hat) + epsilon);
        if (zero_grads) {
            g *= 0.0f;
        }
    }
};

__global__ void adam_step_base_kernel(Gaussians gaussians, Adam adam, bool update_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gaussians.count) {
        return;
    }

    if (update_channels) {
        adam(gaussians.channels[idx], gaussians.first_moment_channels[idx], gaussians.second_moment_channels[idx],
             gaussians.grad_channels[idx], *gaussians.lr_channels);
    } else if (adam.zero_grads) {
        gaussians.grad_channels[idx] = make_floatK(0.0f);
    }
    adam(gaussians.opacity[idx], gaussians.first_moment_opacity[idx], gaussians.second_moment_opacity[idx],
         gaussians.grad_opacity[idx], *gaussians.lr_opacity);

    adam(gaussians.mean[idx], gaussians.first_moment_mean[idx], gaussians.second_moment_mean[idx],
         gaussians.grad_mean[idx], *gaussians.lr_mean);
    adam(gaussians.rotation[idx], gaussians.first_moment_rotation[idx], gaussians.second_moment_rotation[idx],
         gaussians.grad_rotation[idx], *gaussians.lr_rotation);
    adam(gaussians.scale[idx], gaussians.first_moment_scale[idx], gaussians.second_moment_scale[idx],
         gaussians.grad_scale[idx], *gaussians.lr_scale);
}

__global__ void adam_step_sh_kernel(Gaussians gaussians, Adam adam) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gaussians.count * (gaussians.num_sh_coeffs + 1)) {
        return;
    }

    int gaussian_id = idx / (gaussians.num_sh_coeffs + 1);
    int coeff_id = idx % (gaussians.num_sh_coeffs + 1);

    if (coeff_id == 0) {
        adam(gaussians.sh_coeffs_dc[gaussian_id], gaussians.first_moment_sh_coeffs_dc[gaussian_id],
             gaussians.second_moment_sh_coeffs_dc[gaussian_id], gaussians.grad_sh_coeffs_dc[gaussian_id],
             *gaussians.lr_sh_dc);
    } else {
        int k = gaussians.num_sh_coeffs;
        adam(gaussians.sh_coeffs_rest[gaussian_id * k + coeff_id - 1],
             gaussians.first_moment_sh_coeffs_rest[gaussian_id * k + coeff_id - 1],
             gaussians.second_moment_sh_coeffs_rest[gaussian_id * k + coeff_id - 1],
             gaussians.grad_sh_coeffs_rest[gaussian_id * k + coeff_id - 1], *gaussians.lr_sh_rest);
    }
}

void adam_step(Gaussians gaussians, int step, bool zero_grads, bool update_channels, float beta_1, float beta_2,
               float epsilon, bool enable_sh, int sh_update_laziness) {
    Adam adam{step, zero_grads, beta_1, beta_2, epsilon};
    int threads = 256; // * Tweaking this does not seem to affect performance much

    int blocks = (gaussians.count + threads - 1) / threads;
    adam_step_base_kernel<<<blocks, threads, 0>>>(gaussians, adam, update_channels);

    if (enable_sh && step % sh_update_laziness == 0) {
        int blocks2 = (gaussians.count * (gaussians.num_sh_coeffs + 1) + threads - 1) / threads;
        adam_step_sh_kernel<<<blocks2, threads, 1>>>(gaussians, adam);
    }

    cudaDeviceSynchronize();
}