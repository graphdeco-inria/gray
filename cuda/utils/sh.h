#pragma once

#include "../core/gaussians.h"
#include "../core/camera.h"

#include "vec_math.h"
#include <cstdint>
#include <torch/extension.h>
#include <ATen/ATen.h>

using namespace at;

#if CHANNELS == 3
void sh_forward_pass(Gaussians gaussians, Camera camera);
void sh_backward_pass(Gaussians gaussians, Camera camera);
#endif