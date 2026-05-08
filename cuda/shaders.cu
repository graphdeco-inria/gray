#include "params.h"
#include "utils/common.h"
#include "utils/helpers.cu"
#include "core/framebuffer.h" // for floatK

#include "forward_pass.cu"
#include "backward_pass.cu"

extern "C" __global__ void __intersection__is() {
    // * Fetch config
    bool grads_enabled = (bool)optixGetPayload_1();
    float alpha_threshold = __uint_as_float(optixGetPayload_2());
    float exp_power = __uint_as_float(optixGetPayload_3());

    // * Fetch ray data
    float3 local_origin = optixGetObjectRayOrigin();
    float3 local_direction = optixGetObjectRayDirection();

    // * Load gaussian data
    uint32_t gaussian_id = optixGetInstanceIndex();
    float opacity = read_opacity(params, gaussian_id);

    // * Compute pixel index
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();
    uint32_t pixel_id = idx.y * params.image_width + idx.x;

    // * Reject gaussians behind ray
    if (dot(local_origin, local_direction) > 0.0) {
        return;
    }

    // * Compute the hit point along the ray
    float norm = length(local_direction);
    local_direction /= norm;
    float local_hit_distance_along_ray = dot(-local_origin, local_direction);
    float3 local_hit_unscaled = local_origin + local_hit_distance_along_ray * local_direction;

    // * Clip the gaussian at the alpha threshold
    float sq_dist = dot(local_hit_unscaled, local_hit_unscaled);
    if (sq_dist > 1.0f) {
        return;
    }

    // * Compute alpha value
    float3 local_hit = local_hit_unscaled * compute_scaling_factor(opacity, alpha_threshold, exp_power);
    float gaussval = eval_gaussian(local_hit, exp_power);
    float alpha = compute_alpha(gaussval, opacity, alpha_threshold);

    // * Compute the exact total transmittance for the ray
    float full_transmittance = __uint_as_float(optixGetPayload_0());
    full_transmittance *= 1.0 - alpha;
    optixSetPayload_0(__float_as_uint(full_transmittance));

    // * Log all hits to per-pixel linked list
    float distance = local_hit_distance_along_ray / norm;
    params.ppll_forward.insert(grads_enabled, pixel_id, gaussian_id, distance, alpha);

    // * Update stats
    if (grads_enabled) {
        params.stats.num_gaussians_hit[pixel_id]++;
    }
}

extern "C" __global__ void __raygen__rg() {
    // * Compute pixel index
    uint3 idx = optixGetLaunchIndex();
    uint3 dim = optixGetLaunchDimensions();
    uint32_t pixel_id = idx.y * params.image_width + idx.x;

    // * Update random seed based on iteration
    uint32_t seed = tea<4>(pixel_id, *params.metadata.total_num_calls);

    if (*params.metadata.run_forward_pass) {
        // * Compute the ray coordinates
        float3 ray_origin =
            *params.config.rays_from_python ? params.framebuffer.ray_origin[pixel_id] : *params.camera.origin;
        float3 ray_direction =
            *params.config.rays_from_python
                ? params.framebuffer.ray_direction[pixel_id]
                : params.camera.compute_primary_ray_direction(*params.config.jitter_primary_rays, idx, dim, seed);
        if (length(ray_direction) == 0.0f) {
            return; // * Inactive pixel
        }

        // *** Forward pass
        Pixel pixel = forward_pass(pixel_id, ray_origin, ray_direction);

        params.framebuffer.write(pixel_id, pixel);
    }

    // *** Backward pass
    if (*params.metadata.run_backward_pass) {
        // * Fetch the pixel data
        Pixel pixel = params.framebuffer.read(pixel_id);
        backward_pass(pixel_id, pixel);
    }
}
