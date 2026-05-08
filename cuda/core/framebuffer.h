#pragma once

#ifdef __CUDACC__
#include "../utils/common.h"
#endif

#include "../utils/vec_math.h"
#include "gaussians.h" 

struct Pixel {
    floatK output_channels;
    float output_depth;
    float transmittance;
    float full_transmittance;
    floatK remaining_channels_estimate;
    float3 ray_origin;
    float3 ray_direction;
};

struct Framebuffer {
    floatK *__restrict__ output_channels;
    float *__restrict__ output_depth;
    float *__restrict__ transmittance;
    float *__restrict__ full_transmittance;
    floatK *__restrict__ remaining_channels_estimate;
    float3 *__restrict__ ray_origin;
    float3 *__restrict__ ray_direction;

    floatK *__restrict__ grad_output_channels;

#ifdef __CUDACC__
    __device__ void write(uint32_t pixel_id, const Pixel &pixel) {
        output_channels[pixel_id] = pixel.output_channels;
        output_depth[pixel_id] = pixel.output_depth;
        transmittance[pixel_id] = pixel.transmittance;
        full_transmittance[pixel_id] = pixel.full_transmittance;
        remaining_channels_estimate[pixel_id] = pixel.remaining_channels_estimate;
        ray_origin[pixel_id] = pixel.ray_origin;
        ray_direction[pixel_id] = pixel.ray_direction;
    }

    __device__ Pixel read(uint32_t pixel_id) const {
        Pixel pixel;
        pixel.output_channels = output_channels[pixel_id];
        output_depth[pixel_id] = pixel.output_depth;
        pixel.transmittance = transmittance[pixel_id];
        pixel.full_transmittance = full_transmittance[pixel_id];
        pixel.remaining_channels_estimate = remaining_channels_estimate[pixel_id];
        pixel.ray_origin = ray_origin[pixel_id];
        pixel.ray_direction = ray_direction[pixel_id];
        return pixel;
    }
#endif
};

#ifndef __CUDACC__
#include "headers.h"

struct FramebufferDataHolder : torch::CustomClassHolder {
    Tensor output_channels;
    Tensor output_depth;
    Tensor transmittance;
    Tensor full_transmittance;
    Tensor remaining_channels_estimate;
    Tensor ray_origin;
    Tensor ray_direction;

    Tensor grad_output_channels;

    FramebufferDataHolder(uint32_t image_width, uint32_t image_height) {
        output_channels = torch::zeros({image_height, image_width, CHANNELS}, CUDA_FLOAT32);
        output_depth = torch::zeros({image_height, image_width, 1}, CUDA_FLOAT32);
        transmittance = torch::zeros({image_height, image_width, 1}, CUDA_FLOAT32);
        full_transmittance = torch::zeros({image_height, image_width, 1}, CUDA_FLOAT32);
        remaining_channels_estimate = torch::zeros({image_height, image_width, CHANNELS}, CUDA_FLOAT32);
        ray_origin = torch::zeros({image_height, image_width, 3}, CUDA_FLOAT32);
        ray_direction = torch::zeros({image_height, image_width, 3}, CUDA_FLOAT32);

        grad_output_channels = torch::zeros({image_height, image_width, CHANNELS}, CUDA_FLOAT32);
    }

    Framebuffer reify() {
        return Framebuffer{.output_channels = reinterpret_cast<floatK *>(output_channels.data_ptr()),
                           .output_depth = reinterpret_cast<float *>(output_depth.data_ptr()),
                           .transmittance = reinterpret_cast<float *>(transmittance.data_ptr()),
                           .full_transmittance = reinterpret_cast<float *>(full_transmittance.data_ptr()),
                           .remaining_channels_estimate =
                               reinterpret_cast<floatK *>(remaining_channels_estimate.data_ptr()),
                           .ray_origin = reinterpret_cast<float3 *>(ray_origin.data_ptr()),
                           .ray_direction = reinterpret_cast<float3 *>(ray_direction.data_ptr()),

                           .grad_output_channels = reinterpret_cast<floatK *>(grad_output_channels.data_ptr())};
    }

    static void bind(torch::Library &m) {
        m.class_<FramebufferDataHolder>("Framebuffer")
            .def_readonly("output_channels", &FramebufferDataHolder::output_channels)
            .def_readonly("output_depth", &FramebufferDataHolder::output_depth)

            .def_readonly("transmittance", &FramebufferDataHolder::transmittance)
            .def_readonly("full_transmittance", &FramebufferDataHolder::full_transmittance)
            .def_readonly("remaining_channels_estimate", &FramebufferDataHolder::remaining_channels_estimate)

            .def_readonly("ray_origin", &FramebufferDataHolder::ray_origin)
            .def_readonly("ray_direction", &FramebufferDataHolder::ray_direction)

            .def_readonly("grad_output_channels", &FramebufferDataHolder::grad_output_channels);
    }
};

#endif
