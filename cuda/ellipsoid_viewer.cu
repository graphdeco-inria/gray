__device__ __forceinline__ void raygen_ellipsoids(uint32_t pixel_id,
                                                  const float3 ray_origin,
                                                  const float3 ray_direction) {
    params.framebuffer.output_channels[pixel_id] = *params.config.background_channels;
    params.framebuffer.output_depth[pixel_id] = 0.0f;

    const float near_plane = *params.camera.znear;
    const float far_plane = *params.camera.zfar;

    uint32_t dummy = 0, nx_bits = 0, ny_bits = 0, nz_bits = 0;
    optixTrace(params.bvh_handle, ray_origin, ray_direction,
               near_plane, far_plane,
               0.0f,                    // rayTime
               OptixVisibilityMask(1),
               OPTIX_RAY_FLAG_NONE,
               1,                       // SBTOffset: ellipsoid hit group
               0,                       // SBTStride
               0,                       // missSBTIndex
               dummy, nx_bits, ny_bits, nz_bits);
}

extern "C" __global__ void __intersection__is_ellipsoid() {
    // * Skip low-opacity gaussians
    uint32_t gaussian_id = optixGetInstanceIndex();
    if (read_opacity(params, gaussian_id) < *params.config.ellipsoid_min_opacity) {
        return;
    }

    float3 local_o = optixGetObjectRayOrigin();
    float3 local_d = optixGetObjectRayDirection();

    // * Reject gaussians behind ray
    if (dot(local_o, local_d) > 0.0f) {
        return;
    }

    // * Stable sphere intersection via closest-approach (avoids quadratic cancellation)
    float norm = length(local_d);
    float3 local_d_n = local_d / norm;
    float s_mid = dot(-local_o, local_d_n);
    float sq_dist = dot(local_o + s_mid * local_d_n, local_o + s_mid * local_d_n);
    if (sq_dist > 1.0f) {
        return;
    }

    // * Pick nearest visible hit
    float half_chord = sqrtf(1.0f - sq_dist);
    float t0 = (s_mid - half_chord) / norm;
    float t1 = (s_mid + half_chord) / norm;
    float t_min = optixGetRayTmin();
    float t_max = optixGetRayTmax();
    float t_hit;
    if (t0 >= t_min && t0 <= t_max) {
        t_hit = t0;
    } else if (t1 >= t_min && t1 <= t_max) {
        t_hit = t1;
    } else {
        return;
    }

    // * Compute world normal (object-space builtins only available in IS)
    float3 local_normal = normalize(local_o + t_hit * norm * local_d_n);
    float3 world_normal = normalize(optixTransformNormalFromObjectToWorldSpace(local_normal));

    // * Forward world normal to CH via payload slots 1-3
    optixSetPayload_1(__float_as_uint(world_normal.x));
    optixSetPayload_2(__float_as_uint(world_normal.y));
    optixSetPayload_3(__float_as_uint(world_normal.z));
    optixReportIntersection(t_hit, 0);
}

extern "C" __global__ void __closesthit__ellipsoid() {
    // * Compute pixel index
    uint3 idx = optixGetLaunchIndex();
    uint32_t pixel_id = idx.y * params.image_width + idx.x;

    uint32_t gaussian_id = optixGetInstanceIndex();
    float t_hit = optixGetRayTmax();

    // * Recover world normal from payload slots 1-3
    float3 world_normal = make_float3(__uint_as_float(optixGetPayload_1()),
                                      __uint_as_float(optixGetPayload_2()),
                                      __uint_as_float(optixGetPayload_3()));

    // * Flat shading
    float3 view_direction = normalize(optixGetWorldRayDirection());
    float align = fmaxf(0.5f, dot(-view_direction, world_normal));

    floatK gaussian_channels = read_channels(params, gaussian_id);
    floatK output_channels = align * (gaussian_channels + make_floatK(0.15f));

    params.framebuffer.output_channels[pixel_id] = output_channels;
    params.framebuffer.output_depth[pixel_id] = t_hit;
}
