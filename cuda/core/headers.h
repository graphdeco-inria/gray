#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
using namespace at;

#define CUDA_BOOL torch::dtype(torch::kBool).device(torch::kCUDA)
#define CUDA_INT32 torch::dtype(torch::kInt32).device(torch::kCUDA)
#define CUDA_FLOAT32 torch::dtype(torch::kFloat32).device(torch::kCUDA)