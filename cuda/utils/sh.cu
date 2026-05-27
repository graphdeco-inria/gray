#include "sh.h"
#include "complex_math.h"

namespace {

constexpr float kShC0 = 0.28209479177387814f;
constexpr float kShSqrt2 = 1.4142135623730951f;

constexpr int kMaxShDegree = 3;

template <int Degree, typename SH3>
__device__ __forceinline__ float3 evaluate_color_degree(
    float3 dir,
    float3 result,
    const SH3 *coefficients
) {
    constexpr int kRestCoeffCount = (Degree + 1) * (Degree + 1) - 1;
    constexpr int kComplexTerms = (Degree + 1) * (Degree + 2) / 2;

    HarmonicComplex values[kComplexTerms];
    values[complex_term_index(0, 0)] = make_complex_value(kShC0, 0.0f);

    const HarmonicComplex xy = make_complex_value(dir.x, dir.y);

    #pragma unroll
    for (int degree = 1; degree <= Degree; ++degree) {
        int diagonal = complex_term_index(degree, degree);
        int previous_diagonal = complex_term_index(degree - 1, degree - 1);
        float diagonal_scale = -sqrtf((2.0f * degree + 1.0f) / (2.0f * degree));
        values[diagonal] = scale_complex(mul_complex(xy, values[previous_diagonal]), diagonal_scale);

        int subdiagonal = complex_term_index(degree, degree - 1);
        float subdiagonal_scale = sqrtf(2.0f * degree + 1.0f);
        values[subdiagonal] = scale_complex(values[previous_diagonal], subdiagonal_scale * dir.z);

        #pragma unroll
        for (int order = degree - 2; order >= 0; --order) {
            int current = complex_term_index(degree, order);
            int prev = complex_term_index(degree - 1, order);
            int prev_prev = complex_term_index(degree - 2, order);
            float denominator = float(degree * degree - order * order);
            float a = sqrtf((4.0f * degree * degree - 1.0f) / denominator);
            float b = -sqrtf(
                ((2.0f * degree + 1.0f) * (((degree - 1.0f) * (degree - 1.0f)) - order * order)) /
                ((2.0f * degree - 3.0f) * denominator)
            );
            values[current] = add_complex(
                scale_complex(values[prev], a * dir.z),
                scale_complex(values[prev_prev], b)
            );
        }
    }

    float basis[kRestCoeffCount];
    int basis_index = 0;
    #pragma unroll
    for (int degree = 1; degree <= Degree; ++degree) {
        #pragma unroll
        for (int order = degree; order >= 1; --order) {
            basis[basis_index++] = kShSqrt2 * values[complex_term_index(degree, order)].imag;
        }
        basis[basis_index++] = values[complex_term_index(degree, 0)].real;
        #pragma unroll
        for (int order = 1; order <= degree; ++order) {
            basis[basis_index++] = kShSqrt2 * values[complex_term_index(degree, order)].real;
        }
    }

    #pragma unroll
    for (int coeff = 0; coeff < kRestCoeffCount; ++coeff) {
        float3 c = to_float3(coefficients[coeff]);
        float b = basis[coeff];
        result.x = fmaf(b, c.x, result.x);
        result.y = fmaf(b, c.y, result.y);
        result.z = fmaf(b, c.z, result.z);
    }
    return result;
}

template <int Degree>
__device__ __forceinline__ void evaluate_gradients_degree(
    float3 dir,
    const float3 *coefficients,
    float3 *grad_coefficients,
    float3 &rgb_dx,
    float3 &rgb_dy,
    float3 &rgb_dz,
    float3 grad_rgb
) {
    constexpr int kRestCoeffCount = (Degree + 1) * (Degree + 1) - 1;
    constexpr int kComplexTerms = (Degree + 1) * (Degree + 2) / 2;

    HarmonicComplex values[kComplexTerms];
    HarmonicComplex values_dx[kComplexTerms];
    HarmonicComplex values_dy[kComplexTerms];
    HarmonicComplex values_dz[kComplexTerms];

    values[complex_term_index(0, 0)] = make_complex_value(kShC0, 0.0f);
    values_dx[complex_term_index(0, 0)] = make_complex_value();
    values_dy[complex_term_index(0, 0)] = make_complex_value();
    values_dz[complex_term_index(0, 0)] = make_complex_value();

    const HarmonicComplex xy = make_complex_value(dir.x, dir.y);

    #pragma unroll
    for (int degree = 1; degree <= Degree; ++degree) {
        int diagonal = complex_term_index(degree, degree);
        int previous_diagonal = complex_term_index(degree - 1, degree - 1);
        float diagonal_scale = -sqrtf((2.0f * degree + 1.0f) / (2.0f * degree));

        HarmonicComplex prev = values[previous_diagonal];
        HarmonicComplex prev_dx = values_dx[previous_diagonal];
        HarmonicComplex prev_dy = values_dy[previous_diagonal];
        HarmonicComplex prev_dz = values_dz[previous_diagonal];

        values[diagonal] = scale_complex(mul_complex(xy, prev), diagonal_scale);
        values_dx[diagonal] = scale_complex(add_complex(prev, mul_complex(xy, prev_dx)), diagonal_scale);
        values_dy[diagonal] = scale_complex(
            add_complex(mul_complex_i(prev), mul_complex(xy, prev_dy)),
            diagonal_scale
        );
        values_dz[diagonal] = scale_complex(mul_complex(xy, prev_dz), diagonal_scale);

        int subdiagonal = complex_term_index(degree, degree - 1);
        float subdiagonal_scale = sqrtf(2.0f * degree + 1.0f);
        values[subdiagonal] = scale_complex(prev, subdiagonal_scale * dir.z);
        values_dx[subdiagonal] = scale_complex(prev_dx, subdiagonal_scale * dir.z);
        values_dy[subdiagonal] = scale_complex(prev_dy, subdiagonal_scale * dir.z);
        values_dz[subdiagonal] = scale_complex(
            add_complex(prev, scale_complex(prev_dz, dir.z)),
            subdiagonal_scale
        );

        #pragma unroll
        for (int order = degree - 2; order >= 0; --order) {
            int current = complex_term_index(degree, order);
            int prev_idx = complex_term_index(degree - 1, order);
            int prev_prev_idx = complex_term_index(degree - 2, order);
            float denominator = float(degree * degree - order * order);
            float a = sqrtf((4.0f * degree * degree - 1.0f) / denominator);
            float b = -sqrtf(
                ((2.0f * degree + 1.0f) * (((degree - 1.0f) * (degree - 1.0f)) - order * order)) /
                ((2.0f * degree - 3.0f) * denominator)
            );

            values[current] = add_complex(
                scale_complex(values[prev_idx], a * dir.z),
                scale_complex(values[prev_prev_idx], b)
            );
            values_dx[current] = add_complex(
                scale_complex(values_dx[prev_idx], a * dir.z),
                scale_complex(values_dx[prev_prev_idx], b)
            );
            values_dy[current] = add_complex(
                scale_complex(values_dy[prev_idx], a * dir.z),
                scale_complex(values_dy[prev_prev_idx], b)
            );
            values_dz[current] = add_complex(
                scale_complex(add_complex(values[prev_idx], scale_complex(values_dz[prev_idx], dir.z)), a),
                scale_complex(values_dz[prev_prev_idx], b)
            );
        }
    }

    float basis[kRestCoeffCount];
    float basis_dx[kRestCoeffCount];
    float basis_dy[kRestCoeffCount];
    float basis_dz[kRestCoeffCount];
    int basis_index = 0;
    #pragma unroll
    for (int degree = 1; degree <= Degree; ++degree) {
        #pragma unroll
        for (int order = degree; order >= 1; --order) {
            int index = complex_term_index(degree, order);
            basis[basis_index] = kShSqrt2 * values[index].imag;
            basis_dx[basis_index] = kShSqrt2 * values_dx[index].imag;
            basis_dy[basis_index] = kShSqrt2 * values_dy[index].imag;
            basis_dz[basis_index] = kShSqrt2 * values_dz[index].imag;
            ++basis_index;
        }

        int zero_order = complex_term_index(degree, 0);
        basis[basis_index] = values[zero_order].real;
        basis_dx[basis_index] = values_dx[zero_order].real;
        basis_dy[basis_index] = values_dy[zero_order].real;
        basis_dz[basis_index] = values_dz[zero_order].real;
        ++basis_index;

        #pragma unroll
        for (int order = 1; order <= degree; ++order) {
            int index = complex_term_index(degree, order);
            basis[basis_index] = kShSqrt2 * values[index].real;
            basis_dx[basis_index] = kShSqrt2 * values_dx[index].real;
            basis_dy[basis_index] = kShSqrt2 * values_dy[index].real;
            basis_dz[basis_index] = kShSqrt2 * values_dz[index].real;
            ++basis_index;
        }
    }

    #pragma unroll
    for (int coeff = 0; coeff < kRestCoeffCount; ++coeff) {
        float3 c = coefficients[coeff];
        float b = basis[coeff];
        grad_coefficients[coeff] = make_float3(b * grad_rgb.x, b * grad_rgb.y, b * grad_rgb.z);

        float dx = basis_dx[coeff];
        float dy = basis_dy[coeff];
        float dz = basis_dz[coeff];

        rgb_dx.x = fmaf(dx, c.x, rgb_dx.x);
        rgb_dx.y = fmaf(dx, c.y, rgb_dx.y);
        rgb_dx.z = fmaf(dx, c.z, rgb_dx.z);

        rgb_dy.x = fmaf(dy, c.x, rgb_dy.x);
        rgb_dy.y = fmaf(dy, c.y, rgb_dy.y);
        rgb_dy.z = fmaf(dy, c.z, rgb_dy.z);

        rgb_dz.x = fmaf(dz, c.x, rgb_dz.x);
        rgb_dz.y = fmaf(dz, c.y, rgb_dz.y);
        rgb_dz.z = fmaf(dz, c.z, rgb_dz.z);
    }
}

template <typename SH3>
__global__ void sh_forward_kernel(Gaussians gaussians, Camera camera) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= gaussians.count) {
        return;
    }

    float3 delta = gaussians.mean[index] - *camera.origin;
    float3 dir = safe_normalize(delta);
    float3 result = kShC0 * to_float3(reinterpret_cast<const SH3 *>(gaussians.sh_coeffs_dc)[index]);

    int degree = min(*gaussians.current_sh_degree, kMaxShDegree);
    if (degree > 0) {
        const SH3 *coefficients = reinterpret_cast<const SH3 *>(gaussians.sh_coeffs_rest) + index * gaussians.num_sh_coeffs;
        switch (degree) {
        case 1: result = evaluate_color_degree<1>(dir, result, coefficients); break;
        case 2: result = evaluate_color_degree<2>(dir, result, coefficients); break;
        case 3: result = evaluate_color_degree<3>(dir, result, coefficients); break;
        }
    }

    result += make_float3(0.5f, 0.5f, 0.5f);
    gaussians.channels[index] = make_float3(fmaxf(result.x, 0.0f), fmaxf(result.y, 0.0f), fmaxf(result.z, 0.0f));
}

__global__ void sh_backward_kernel(Gaussians gaussians, Camera camera) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= gaussians.count || !gaussians.was_visible[index]) {
        return;
    }

    float3 delta = gaussians.mean[index] - *camera.origin;
    float3 dir = safe_normalize(delta);

    float3 grad_rgb = gaussians.grad_channels[index];
    float3 channels = gaussians.channels[index];
    if (channels.x <= 0.0f) grad_rgb.x = 0.0f;
    if (channels.y <= 0.0f) grad_rgb.y = 0.0f;
    if (channels.z <= 0.0f) grad_rgb.z = 0.0f;

    gaussians.grad_sh_coeffs_dc[index] = kShC0 * grad_rgb;

    float3 rgb_dx = make_float3(0.0f, 0.0f, 0.0f);
    float3 rgb_dy = make_float3(0.0f, 0.0f, 0.0f);
    float3 rgb_dz = make_float3(0.0f, 0.0f, 0.0f);

    int degree = min(*gaussians.current_sh_degree, kMaxShDegree);
    if (degree > 0) {
        const float3 *coefficients = gaussians.sh_coeffs_rest + index * gaussians.num_sh_coeffs;
        float3 *grad_coefficients = gaussians.grad_sh_coeffs_rest + index * gaussians.num_sh_coeffs;

        switch (degree) {
        case 1: evaluate_gradients_degree<1>(dir, coefficients, grad_coefficients, rgb_dx, rgb_dy, rgb_dz, grad_rgb); break;
        case 2: evaluate_gradients_degree<2>(dir, coefficients, grad_coefficients, rgb_dx, rgb_dy, rgb_dz, grad_rgb); break;
        case 3: evaluate_gradients_degree<3>(dir, coefficients, grad_coefficients, rgb_dx, rgb_dy, rgb_dz, grad_rgb); break;
        }
    }

    float3 grad_dir = make_float3(dot(rgb_dx, grad_rgb), dot(rgb_dy, grad_rgb), dot(rgb_dz, grad_rgb));
    float length_sq = fmaxf(dot(delta, delta), 1e-20f);
    float inv_len = rsqrtf(length_sq);
    float inv_len_cubed = inv_len * inv_len * inv_len;
    gaussians.grad_mean[index] += grad_dir * inv_len - delta * (dot(delta, grad_dir) * inv_len_cubed);
}

} // namespace

void sh_forward_pass(Gaussians gaussians, Camera camera) {
    constexpr int kThreads = 128;
    int blocks = (gaussians.count + kThreads - 1) / kThreads;
    if (gaussians.sh_is_fp16)
        sh_forward_kernel<half3><<<blocks, kThreads>>>(gaussians, camera);
    else
        sh_forward_kernel<float3><<<blocks, kThreads>>>(gaussians, camera);
}

void sh_backward_pass(Gaussians gaussians, Camera camera) {
    constexpr int kThreads = 128;
    int blocks = (gaussians.count + kThreads - 1) / kThreads;
    sh_backward_kernel<<<blocks, kThreads>>>(gaussians, camera);
}
