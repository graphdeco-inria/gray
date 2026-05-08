/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "sh.h"

__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f, -1.0925484305920792f,
                                  0.5462742152960396f};
__device__ const float SH_C3[] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f, 0.3731763325901154f,
                                  -0.4570457994644658f, 1.445305721320277f, -0.5900435899266435f};

__global__ void sh_forward_pass_kernel(Gaussians gaussians, Camera camera) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gaussians.count)
        return;

    // The implementation is loosely based on code for
    // "Differentiable Point-Based Radiance Fields for
    // Efficient View Synthesis" by Zhang et al. (2022)
    float3 pos = gaussians.mean[idx];
    float3 dir = pos - *camera.origin;
    dir = dir / length(dir);

    int max_coeffs = (gaussians.max_sh_degree + 1) * (gaussians.max_sh_degree + 1) - 1;
    const float3 *sh_rest = gaussians.sh_coeffs_rest + idx * max_coeffs;
    floatK result = SH_C0 * gaussians.sh_coeffs_dc[idx];

    int deg = *gaussians.current_sh_degree;

    if (deg > 0) {
        float x = dir.x;
        float y = dir.y;
        float z = dir.z;
        result = result - SH_C1 * y * sh_rest[0] + SH_C1 * z * sh_rest[1] - SH_C1 * x * sh_rest[2];

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result = result + SH_C2[0] * xy * sh_rest[3] + SH_C2[1] * yz * sh_rest[4] +
                     SH_C2[2] * (2.0f * zz - xx - yy) * sh_rest[5] + SH_C2[3] * xz * sh_rest[6] +
                     SH_C2[4] * (xx - yy) * sh_rest[7];

            if (deg > 2) {
                result = result + SH_C3[0] * y * (3.0f * xx - yy) * sh_rest[8] + SH_C3[1] * xy * z * sh_rest[9] +
                         SH_C3[2] * y * (4.0f * zz - xx - yy) * sh_rest[10] +
                         SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh_rest[11] +
                         SH_C3[4] * x * (4.0f * zz - xx - yy) * sh_rest[12] + SH_C3[5] * z * (xx - yy) * sh_rest[13] +
                         SH_C3[6] * x * (xx - 3.0f * yy) * sh_rest[14];
            }
        }
    }
    result += make_floatK(0.5f);

    gaussians.channels[idx] = make_float3(fmaxf(result.x, 0.0f), fmaxf(result.y, 0.0f), fmaxf(result.z, 0.0f));
}

void sh_forward_pass(Gaussians gaussians, Camera camera) {
    sh_forward_pass_kernel<<<(gaussians.count + 255) / 256, 256>>>(gaussians, camera);
}

__device__ float3 dnormvdv(float3 v, float3 dv) {
    float sum2 = v.x * v.x + v.y * v.y + v.z * v.z;
    float invsum32 = 1.0f / sqrtf(sum2 * sum2 * sum2);

    float3 dnormvdv;
    dnormvdv.x = ((+sum2 - v.x * v.x) * dv.x - v.y * v.x * dv.y - v.z * v.x * dv.z) * invsum32;
    dnormvdv.y = (-v.x * v.y * dv.x + (sum2 - v.y * v.y) * dv.y - v.z * v.y * dv.z) * invsum32;
    dnormvdv.z = (-v.x * v.z * dv.x - v.y * v.z * dv.y + (sum2 - v.z * v.z) * dv.z) * invsum32;
    return dnormvdv;
}

__global__ void sh_backward_pass_kernel(Gaussians gaussians, Camera camera) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gaussians.count)
        return;

    // * Skip gaussians that were not visible in the forward pass
    if (!gaussians.was_visible[idx]) {
        return;
    }

    float3 pos = gaussians.mean[idx];
    float3 dir_orig = pos - *camera.origin;
    float3 dir = dir_orig / length(dir_orig);

    int max_coeffs = (gaussians.max_sh_degree + 1) * (gaussians.max_sh_degree + 1) - 1;
    const float3 *sh_rest = gaussians.sh_coeffs_rest + idx * max_coeffs;
    floatK dL_dRGB = gaussians.grad_channels[idx];

    float3 channel_values = gaussians.channels[idx];
    if (channel_values.x < 0.0f)
        dL_dRGB.x *= 0.0f;
    if (channel_values.y < 0.0f)
        dL_dRGB.y *= 0.0f;
    if (channel_values.z < 0.0f)
        dL_dRGB.z *= 0.0f;

    floatK dRGBdx = make_floatK(0.0f);
    floatK dRGBdy = make_floatK(0.0f);
    floatK dRGBdz = make_floatK(0.0f);
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;

    float3 *dL_dsh_rest = gaussians.grad_sh_coeffs_rest + idx * max_coeffs;
    int deg = *gaussians.current_sh_degree;

    float dRGBdsh0 = SH_C0;
    gaussians.grad_sh_coeffs_dc[idx] = dRGBdsh0 * dL_dRGB;

    if (deg > 0) {
        float dRGBdsh1 = -SH_C1 * y;
        float dRGBdsh2 = SH_C1 * z;
        float dRGBdsh3 = -SH_C1 * x;
        dL_dsh_rest[0] = dRGBdsh1 * dL_dRGB;
        dL_dsh_rest[1] = dRGBdsh2 * dL_dRGB;
        dL_dsh_rest[2] = dRGBdsh3 * dL_dRGB;

        dRGBdx = -SH_C1 * sh_rest[2];
        dRGBdy = -SH_C1 * sh_rest[0];
        dRGBdz = SH_C1 * sh_rest[1];

        if (deg > 1) {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;

            float dRGBdsh4 = SH_C2[0] * xy;
            float dRGBdsh5 = SH_C2[1] * yz;
            float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
            float dRGBdsh7 = SH_C2[3] * xz;
            float dRGBdsh8 = SH_C2[4] * (xx - yy);
            dL_dsh_rest[3] = dRGBdsh4 * dL_dRGB;
            dL_dsh_rest[4] = dRGBdsh5 * dL_dRGB;
            dL_dsh_rest[5] = dRGBdsh6 * dL_dRGB;
            dL_dsh_rest[6] = dRGBdsh7 * dL_dRGB;
            dL_dsh_rest[7] = dRGBdsh8 * dL_dRGB;

            dRGBdx += SH_C2[0] * y * sh_rest[3] + SH_C2[2] * 2.f * -x * sh_rest[5] + SH_C2[3] * z * sh_rest[6] +
                      SH_C2[4] * 2.f * x * sh_rest[7];
            dRGBdy += SH_C2[0] * x * sh_rest[3] + SH_C2[1] * z * sh_rest[4] + SH_C2[2] * 2.f * -y * sh_rest[5] +
                      SH_C2[4] * 2.f * -y * sh_rest[7];
            dRGBdz += SH_C2[1] * y * sh_rest[4] + SH_C2[2] * 2.f * 2.f * z * sh_rest[5] + SH_C2[3] * x * sh_rest[6];

            if (deg > 2) {
                float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
                float dRGBdsh10 = SH_C3[1] * xy * z;
                float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
                float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
                float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
                float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
                float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
                dL_dsh_rest[8] = dRGBdsh9 * dL_dRGB;
                dL_dsh_rest[9] = dRGBdsh10 * dL_dRGB;
                dL_dsh_rest[10] = dRGBdsh11 * dL_dRGB;
                dL_dsh_rest[11] = dRGBdsh12 * dL_dRGB;
                dL_dsh_rest[12] = dRGBdsh13 * dL_dRGB;
                dL_dsh_rest[13] = dRGBdsh14 * dL_dRGB;
                dL_dsh_rest[14] = dRGBdsh15 * dL_dRGB;

                dRGBdx += (SH_C3[0] * sh_rest[8] * 3.f * 2.f * xy + SH_C3[1] * sh_rest[9] * yz +
                           SH_C3[2] * sh_rest[10] * -2.f * xy + SH_C3[3] * sh_rest[11] * -3.f * 2.f * xz +
                           SH_C3[4] * sh_rest[12] * (-3.f * xx + 4.f * zz - yy) + SH_C3[5] * sh_rest[13] * 2.f * xz +
                           SH_C3[6] * sh_rest[14] * 3.f * (xx - yy));

                dRGBdy += (SH_C3[0] * sh_rest[8] * 3.f * (xx - yy) + SH_C3[1] * sh_rest[9] * xz +
                           SH_C3[2] * sh_rest[10] * (-3.f * yy + 4.f * zz - xx) +
                           SH_C3[3] * sh_rest[11] * -3.f * 2.f * yz + SH_C3[4] * sh_rest[12] * -2.f * xy +
                           SH_C3[5] * sh_rest[13] * -2.f * yz + SH_C3[6] * sh_rest[14] * -3.f * 2.f * xy);

                dRGBdz += (SH_C3[1] * sh_rest[9] * xy + SH_C3[2] * sh_rest[10] * 4.f * 2.f * yz +
                           SH_C3[3] * sh_rest[11] * 3.f * (2.f * zz - xx - yy) +
                           SH_C3[4] * sh_rest[12] * 4.f * 2.f * xz + SH_C3[5] * sh_rest[13] * (xx - yy));
            }
        }
    }

    float3 dL_ddir = make_float3(dot(dRGBdx, dL_dRGB), dot(dRGBdy, dL_dRGB), dot(dRGBdz, dL_dRGB));
    float3 dL_dmean = dnormvdv(dir_orig, dL_ddir);
    gaussians.grad_mean[idx] += dL_dmean;
}

void sh_backward_pass(Gaussians gaussians, Camera camera) {
    sh_backward_pass_kernel<<<(gaussians.count + 255) / 256, 256>>>(gaussians, camera);
}
