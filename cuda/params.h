#pragma once

#include <iostream>
#include <random>
#include <cuda.h>
#include <optix.h>

#include "utils/vec_math.h"

#include "core/all.h"

struct Params {
    uint32_t image_width;
    uint32_t image_height;

    Camera camera;
    Config config;
    Framebuffer framebuffer;
    Gaussians gaussians;
    Metadata metadata;
    Stats stats;

    OptixTraversableHandle bvh_handle;

    PerPixelLinkedList ppll_forward;
    PerPixelLinkedList ppll_backward;
};

extern "C" {
__constant__ Params params;
}
