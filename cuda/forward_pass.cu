#include <tuple>

constexpr int MAX_ITERATIONS = 99;
constexpr int BUFFER_SIZE = 32;

__device__ __forceinline__ Pixel forward_pass(uint32_t pixel_id, const float3 ray_origin, const float3 ray_direction) {
    const bool grads_enabled = *params.metadata.grads_enabled;
    const bool render_depth = *params.config.render_depth;
    const float t_threshold = *params.config.t_threshold;

    const float near_plane = *params.camera.znear;
    const float far_plane = *params.camera.zfar;

    // * Traverse BVH
    uint32_t full_transmittance_uint = __float_as_uint(1.0f);
    uint32_t grads_enabled_uint = *params.metadata.grads_enabled;
    uint32_t alpha_threshold_uint = __float_as_uint(*params.config.alpha_threshold);
    uint32_t exp_power_uint = __float_as_uint(*params.config.exp_power);
    optixTraverse(params.bvh_handle, ray_origin, ray_direction,
                  near_plane, // tmin
                  far_plane,
                  0.0f, // rayTime
                  OptixVisibilityMask(1), OPTIX_RAY_FLAG_NONE,
                  0, // SBTOffset
                  0, // SBTStride
                  0, // missSBTIndex
                  full_transmittance_uint, grads_enabled_uint, alpha_threshold_uint, exp_power_uint);
    float transmittance = 1.0f;
    float full_transmittance = __uint_as_float(full_transmittance_uint);

    // * Initialize registers holding the BUFFER_SIZE nearest gaussians
    float distances[BUFFER_SIZE];
    unsigned int hit_idxes[BUFFER_SIZE];
    floatK output_channels = make_floatK(0.0f);
    float output_depth = 0.0f;

    // * Variables to estimate the contribution of truncated gaussians
    floatK skipped_weighted_channels = make_floatK(0.0f);
    float skipped_total_alpha = 0.0f;

    // * Stats
    int num_gaussians_accumulated = 0;

    // * Loop over batches from the PPLL
    float tmin = near_plane;
    for (int iteration = 0; iteration < MAX_ITERATIONS && tmin < far_plane; iteration++) {
        fill_array(distances, BUFFER_SIZE, std::numeric_limits<float>::max());
        fill_array(hit_idxes, BUFFER_SIZE, PerPixelLinkedList::NULL_PTR);

        // * Fill batch with nearest gaussians behind the last one
        for (auto hit_idx : params.ppll_forward.pixel_view(pixel_id)) {
            // * In the first iteration, accumulate values over all gaussians for estimating the truncated contribution
            if (iteration == 0) {
                float alpha = params.ppll_forward.alphas[hit_idx];
                skipped_weighted_channels += alpha * read_channels(params, params.ppll_forward.gaussian_ids[hit_idx]);
                skipped_total_alpha += alpha;
            }

            float curr_distance = params.ppll_forward.distances[hit_idx];
            if (curr_distance > tmin && curr_distance < distances[BUFFER_SIZE - 1]) {
                distances[BUFFER_SIZE - 1] = curr_distance;
                hit_idxes[BUFFER_SIZE - 1] = hit_idx;
            }
#pragma unroll
            for (int i = BUFFER_SIZE - 1; i > 0; i--) {
                if (distances[i] < distances[i - 1]) {
                    // * Swap i with i-1
                    float tmp_dist = distances[i];
                    int tmp_idx = hit_idxes[i];
                    distances[i] = distances[i - 1];
                    hit_idxes[i] = hit_idxes[i - 1];
                    distances[i - 1] = tmp_dist;
                    hit_idxes[i - 1] = tmp_idx;
                }
            }
        }

        // * Break if all gaussians are processed
        if (hit_idxes[0] == PerPixelLinkedList::NULL_PTR) {
            break;
        }

// * Integrate the batch of gaussians
#pragma unroll
        for (int i = 0; i < BUFFER_SIZE; i++) {
            float distance = distances[i];
            tmin = max(distance, tmin);

            if (distance < far_plane) {
                // * Update stats
                num_gaussians_accumulated++;

                // * Fetch data from PPLL
                uint32_t gaussian_id = params.ppll_forward.gaussian_ids[hit_idxes[i]];
                float alpha = params.ppll_forward.alphas[hit_idxes[i]];

                // * Fetch gaussian data
                floatK gaussian_channels = read_channels(params, gaussian_id);

                // * Remove integrated gaussians from truncated contribution estimate
                skipped_weighted_channels -= alpha * gaussian_channels;
                skipped_total_alpha -= alpha;

                // * Accumulate values
                float weight = transmittance * alpha;
                output_channels += gaussian_channels * weight;
                if (render_depth) {
                    output_depth += distance * weight;
                }
                transmittance = transmittance * (1.0f - alpha);

                // * Store data required in backward pass PPLL
                if (grads_enabled) {
                    params.ppll_backward.insert(grads_enabled, pixel_id, gaussian_id, distance, alpha);
                }

                // * Break if transmittance threshold is reached
                if (transmittance < t_threshold) {
                    tmin = far_plane;
                    break;
                }
            }
        }
    }

    // * Writeout stats
    if (grads_enabled) {
        params.stats.num_gaussians_accumulated[pixel_id] += num_gaussians_accumulated;
    }

    // * Approximate the contribution of truncated gaussians
    float remaining_transmittance = transmittance - full_transmittance;
    floatK remaining_channels_estimate =
        skipped_total_alpha > 0.0f ? (skipped_weighted_channels / skipped_total_alpha) : make_floatK(0.0f);
    output_channels += remaining_transmittance * remaining_channels_estimate;
    if (render_depth) {
        float normalization = max((1.0f - transmittance), *params.config.eps_forward_normalization);
        output_depth = output_depth / normalization;
    }

    // * Background channels
    output_channels += *params.config.background_channels * full_transmittance;

    return Pixel{.output_channels = output_channels,
                 .output_depth = output_depth,
                 .transmittance = transmittance,
                 .full_transmittance = full_transmittance,
                 .remaining_channels_estimate = remaining_channels_estimate,
                 .ray_origin = ray_origin,
                 .ray_direction = ray_direction};
}
