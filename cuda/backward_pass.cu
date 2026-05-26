
__device__ __forceinline__ void backward_pass(const uint32_t pixel_id, const Pixel &pixel) {
    floatK background_channels = *params.config.background_channels;
    floatK remaining_channels_estimate = pixel.remaining_channels_estimate;

    // * Preload config parameters
    const float alpha_threshold = *params.config.alpha_threshold;
    const float exp_power = *params.config.exp_power;
    const float eps_scale_grad = *params.config.eps_scale_grad;

    // * Init variables used to flow gradient from back to front
    float grad_transmittance = 0.0f;
    float transmittance = pixel.transmittance;

    for (auto hit_idx : params.ppll_backward.pixel_view(pixel_id)) {
        // * Read all PPLL data
        uint32_t gaussian_id = params.ppll_backward.gaussian_ids[hit_idx];

        // * Read all gaussian data
        float opacity = read_opacity(params, gaussian_id);
        float3 scaling = read_scale(params, gaussian_id);
        float4 rotation_unnormalized = params.gaussians.rotation[gaussian_id];
        float4 rotation = normalize_act(rotation_unnormalized);
        floatK gaussian_channels = read_channels(params, gaussian_id);

        // * Fetch the transform matrices
        const float4 *world_to_local = optixGetInstanceInverseTransformFromHandle(
            optixGetInstanceTraversableFromIAS(params.bvh_handle, gaussian_id));
        const float4 *local_to_world =
            optixGetInstanceTransformFromHandle(optixGetInstanceTraversableFromIAS(params.bvh_handle, gaussian_id));

        // * Recompute the local hit point
        float3 local_ray_origin = make_float3(dot(world_to_local[0], make_float4(pixel.ray_origin, 1.0f)),
                                              dot(world_to_local[1], make_float4(pixel.ray_origin, 1.0f)),
                                              dot(world_to_local[2], make_float4(pixel.ray_origin, 1.0f)));
        float3 local_ray_direction = make_float3(dot(make_float3(world_to_local[0]), pixel.ray_direction),
                                                 dot(make_float3(world_to_local[1]), pixel.ray_direction),
                                                 dot(make_float3(world_to_local[2]), pixel.ray_direction));
        float norm = length(local_ray_direction);
        local_ray_direction /= norm;
        float local_hit_distance_along_ray = dot(-local_ray_origin, local_ray_direction);
        float3 local_hit_unscaled = local_ray_origin + local_hit_distance_along_ray * local_ray_direction;
        float3 local_hit = local_hit_unscaled * compute_scaling_factor(opacity, alpha_threshold, exp_power);
        float gaussval = eval_gaussian(local_hit, exp_power);
        float alpha = compute_alpha(gaussval, opacity, alpha_threshold); // recompute for stability

        // * Output buffer gradient
        floatK grad_output_channels = params.framebuffer.grad_output_channels[pixel_id];

        // * Channels gradient
        float weight = transmittance / (1.0f - alpha) * alpha;
        floatK grad_gaussian_channels = backward_act_for_channels(grad_output_channels * weight, gaussian_channels);

        // * Alpha gradient
        float grad_alpha = 0.0f;
        grad_alpha +=
            (dot(grad_output_channels, gaussian_channels) - grad_transmittance) * transmittance / (1.0f - alpha);
        grad_alpha += -pixel.full_transmittance / (1.0f - alpha) * dot(background_channels, grad_output_channels);
        grad_alpha += -((pixel.transmittance - pixel.full_transmittance) / (1.0f - alpha)) *
                      dot(remaining_channels_estimate, grad_output_channels);

        // * Update transmittance gradient
        grad_transmittance += (dot(grad_output_channels, gaussian_channels) - grad_transmittance) * alpha;

        // * Opacity gradient
        float grad_gaussian_opacity = MAX_ALPHA * grad_alpha * gaussval;
        grad_gaussian_opacity = backward_sigmoid_act(grad_gaussian_opacity, opacity);

        // * Transform gradient
        float grad_gaussval = MAX_ALPHA * grad_alpha * opacity;
        float sq_norm = dot(local_hit, local_hit);
        float grad_sq_norm;
        if (exp_power == 1.0f) {
            grad_sq_norm = gaussval;
        } else if (exp_power == 2.0f) {
            grad_sq_norm = gaussval * sq_norm;
        } else if (exp_power == 3.0f) {
            grad_sq_norm = gaussval * sq_norm * sq_norm;
        } else {
            grad_sq_norm = gaussval * powf(sq_norm, exp_power - 1.0f);
        }
        float3 grad_x_local = -local_hit * grad_sq_norm * grad_gaussval;

        // * World hit point gradient
        float scaling_factor = compute_scaling_factor(opacity, alpha_threshold, exp_power);
        float3 grad_x_world =
            make_float3(dot(make_float3(world_to_local[0].x, world_to_local[1].x, world_to_local[2].x), grad_x_local),
                        dot(make_float3(world_to_local[0].y, world_to_local[1].y, world_to_local[2].y), grad_x_local),
                        dot(make_float3(world_to_local[0].z, world_to_local[1].z, world_to_local[2].z), grad_x_local)) *
            scaling_factor;

        // * Local to world matrix gradient
        float3 grad_l2w_0 = -grad_x_world.x * local_hit;
        float3 grad_l2w_1 = -grad_x_world.y * local_hit;
        float3 grad_l2w_2 = -grad_x_world.z * local_hit;

        // * Mean gradient
        float3 grad_gaussian_mean = -grad_x_world;

        // * Scaling gradient
        float3 rot_0 = make_float3(local_to_world[0]) / (scaling * scaling_factor + eps_scale_grad);
        float3 rot_1 = make_float3(local_to_world[1]) / (scaling * scaling_factor + eps_scale_grad);
        float3 rot_2 = make_float3(local_to_world[2]) / (scaling * scaling_factor + eps_scale_grad);
        float3 grad_gaussian_scale =
            backward_exp_act(grad_l2w_0 * rot_0 + grad_l2w_1 * rot_1 + grad_l2w_2 * rot_2, scaling);

        // * Rotation matrix gradient
        float3 grad_rot_0 = grad_l2w_0 * scaling;
        float3 grad_rot_1 = grad_l2w_1 * scaling;
        float3 grad_rot_2 = grad_l2w_2 * scaling;

        // * Rotation quaternion gradient
        float r = rotation.x;
        float x = rotation.y;
        float y = rotation.z;
        float z = rotation.w;
        float grad_r = (2.f * x * (grad_rot_2.y - grad_rot_1.z) + 2.f * y * (grad_rot_0.z - grad_rot_2.x) +
                        2.f * z * (grad_rot_1.x - grad_rot_0.y));
        float grad_x = (-4.f * x * (grad_rot_1.y + grad_rot_2.z) + 2.f * y * (grad_rot_0.y + grad_rot_1.x) +
                        2.f * z * (grad_rot_0.z + grad_rot_2.x) + 2.f * r * (grad_rot_2.y - grad_rot_1.z));
        float grad_y = (2.f * x * (grad_rot_0.y + grad_rot_1.x) - 4.f * y * (grad_rot_0.x + grad_rot_2.z) +
                        2.f * z * (grad_rot_1.z + grad_rot_2.y) + 2.f * r * (grad_rot_0.z - grad_rot_2.x));
        float grad_z = (2.f * x * (grad_rot_0.z + grad_rot_2.x) + 2.f * y * (grad_rot_1.z + grad_rot_2.y) -
                        4.f * z * (grad_rot_0.x + grad_rot_1.y) + 2.f * r * (grad_rot_1.x - grad_rot_0.y));
        float4 grad_gaussian_rotation =
            backward_normalize_act(make_float4(grad_r, grad_x, grad_y, grad_z), rotation_unnormalized, rotation);

        // * Compute the normalized weight used for pruning
        float normalized_weight = weight / float(optixGetLaunchDimensions().x * optixGetLaunchDimensions().y);

        // * Flush to memory
        atomicAddX(&params.gaussians.grad_rotation[gaussian_id], grad_gaussian_rotation);
        atomicAddX(&params.gaussians.grad_scale[gaussian_id], grad_gaussian_scale);
        atomicAddX(&params.gaussians.grad_mean[gaussian_id], grad_gaussian_mean);
        atomicAddX(&params.gaussians.grad_opacity[gaussian_id], grad_gaussian_opacity);
        atomicAddX(&params.gaussians.grad_channels[gaussian_id], grad_gaussian_channels);
        atomicAdd(&params.gaussians.pruning_weight[gaussian_id], normalized_weight);
        params.gaussians.was_visible[gaussian_id] = true;

        // * Update transmittance for next iteration
        transmittance = transmittance / (1.0f - alpha);
    }
}