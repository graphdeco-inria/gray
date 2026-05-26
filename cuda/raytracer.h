#pragma once

#ifdef _WIN32
#define NOMINMAX
#endif

#include <ATen/ATen.h>
#include <cstddef>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <string>
#include <torch/extension.h>
#include <tuple>

#include "utils/exception.h"
#include "core/all.h"
#include "params.h"
#include "optix/pipeline_wrapper.h"
#include "optix/bvh_wrapper.h"
#include "opt/adam.h"
#include "utils/sh.h"

struct Raytracer : torch::CustomClassHolder {
    int width;
    int height;

    Params params_on_host;
    CUdeviceptr params_on_device;

    c10::intrusive_ptr<CameraDataHolder> camera_data;
    c10::intrusive_ptr<ConfigDataHolder> config_data;
    c10::intrusive_ptr<FramebufferDataHolder> framebuffer_data;
    c10::intrusive_ptr<GaussianDataHolder> gaussian_data;
    c10::intrusive_ptr<MetaDataHolder> meta_data;
    c10::intrusive_ptr<StatsDataHolder> stats_data;
    c10::intrusive_ptr<PPLLDataHolder> ppll_forward_data;
    c10::intrusive_ptr<PPLLDataHolder> ppll_backward_data;
    bool training_buffers_enabled;

    std::unique_ptr<PipelineWrapper> pipeline_wrapper;
    std::unique_ptr<BVHWrapper> bvh_wrapper;

        Raytracer(int64_t width_, int64_t height_, int64_t num_gaussians, int64_t sh_max_degree,
                            int64_t ppll_forward_size, int64_t ppll_backward_size, bool training_buffers_enabled_ = true)
        : width(width_), height(height_), camera_data(c10::make_intrusive<CameraDataHolder>()),
          config_data(c10::make_intrusive<ConfigDataHolder>()),
          framebuffer_data(c10::make_intrusive<FramebufferDataHolder>(width, height)),
                    gaussian_data(c10::make_intrusive<GaussianDataHolder>(sh_max_degree, training_buffers_enabled_)),
          meta_data(c10::make_intrusive<MetaDataHolder>(width, height)),
          stats_data(c10::make_intrusive<StatsDataHolder>(width, height)),
          ppll_forward_data(c10::make_intrusive<PPLLDataHolder>("forward\0", width, height, ppll_forward_size)),
                    ppll_backward_data(c10::make_intrusive<PPLLDataHolder>(
                            "backward\0", width, height, training_buffers_enabled_ ? ppll_backward_size : 1)),
                    training_buffers_enabled(training_buffers_enabled_) {
        ppll_forward_data->reset();
        ppll_backward_data->reset();

        params_on_host.image_width = width;
        params_on_host.image_height = height;
        params_on_host.camera = camera_data->reify();
        params_on_host.config = config_data->reify();
        params_on_host.framebuffer = framebuffer_data->reify();
        params_on_host.gaussians = gaussian_data->reify();
        params_on_host.ppll_forward = ppll_forward_data->reify();
        params_on_host.ppll_backward = ppll_backward_data->reify();
        params_on_host.metadata = meta_data->reify();
        params_on_host.stats = stats_data->reify();

        pipeline_wrapper = std::make_unique<PipelineWrapper>();
        bvh_wrapper = std::make_unique<BVHWrapper>(pipeline_wrapper->context, *config_data, params_on_host);

        params_on_host.bvh_handle = bvh_wrapper->tlas_handle;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&params_on_device), sizeof(Params)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(params_on_device), &params_on_host, sizeof(Params),
                              cudaMemcpyHostToDevice));

        if (num_gaussians > 0) {
            resize(num_gaussians);
        }
    }

    void forward_pass() {
        meta_data->update();

        ppll_forward_data->reset();
        ppll_backward_data->reset();
        gaussian_data->was_visible.fill_(false);

        if (config_data->enable_sh.item<bool>()) {
            sh_forward_pass(params_on_host.gaussians, params_on_host.camera);
        }

        meta_data->run_backward_pass.fill_(false);
        pipeline_wrapper->launch(width, height, params_on_device);
        meta_data->run_backward_pass.fill_(true);
    }

    void forward_pass_display() {
        CUDA_CHECK(cudaMemset(ppll_forward_data->total_hits.data_ptr(), 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(ppll_forward_data->head_per_pixel.data_ptr(), 0xff,
                              ppll_forward_data->head_per_pixel.nbytes()));

        if (config_data->enable_sh.item<bool>()) {
            sh_forward_pass(params_on_host.gaussians, params_on_host.camera);
        }

        pipeline_wrapper->launch(width, height, params_on_device);
    }

    void backward_pass() {
        meta_data->run_forward_pass.fill_(false);
        pipeline_wrapper->launch(width, height, params_on_device);
        meta_data->run_forward_pass.fill_(true);

        if (config_data->enable_sh.item<bool>()) {
            sh_backward_pass(params_on_host.gaussians, params_on_host.camera);
        }
    }

    void step() {
        adam_step(params_on_host.gaussians, meta_data->total_num_steps.item<int>(),
                  config_data->zero_grads.item<bool>(), config_data->update_channels.item<bool>(),
                  gaussian_data->beta_1.item<float>(), gaussian_data->beta_2.item<float>(),
                  gaussian_data->epsilon.item<float>(), config_data->enable_sh.item<bool>(),
                  gaussian_data->sh_update_laziness.item<int>());
    }

    void update_bvh() { bvh_wrapper->update(); }

    void rebuild_bvh() {
        bvh_wrapper->rebuild();
        params_on_host.bvh_handle = bvh_wrapper->tlas_handle;
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(params_on_device + offsetof(Params, bvh_handle)),
                              &params_on_host.bvh_handle, sizeof(params_on_host.bvh_handle), cudaMemcpyHostToDevice));
    }

    void resize(int64_t new_num_gaussians) {
        gaussian_data->resize(new_num_gaussians);
        params_on_host.gaussians = gaussian_data->reify();
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(params_on_device + offsetof(Params, gaussians)),
                              &params_on_host.gaussians, sizeof(Gaussians), cudaMemcpyHostToDevice));
    }

    static void bind(torch::Library &m) {
        m.class_<Raytracer>("Raytracer")
            .def(torch::init<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, bool>())
            .def("forward_pass", &Raytracer::forward_pass)
            .def("forward_pass_display", &Raytracer::forward_pass_display)
            .def("backward_pass", &Raytracer::backward_pass)
            .def("step", &Raytracer::step)
            .def("update_bvh", &Raytracer::update_bvh)
            .def("rebuild_bvh", &Raytracer::rebuild_bvh)
            .def("resize", &Raytracer::resize)
            .def("get_camera", [](const c10::intrusive_ptr<Raytracer> &self) { return self->camera_data; })
            .def("get_config", [](const c10::intrusive_ptr<Raytracer> &self) { return self->config_data; })
            .def("get_framebuffer", [](const c10::intrusive_ptr<Raytracer> &self) { return self->framebuffer_data; })
            .def("get_gaussians", [](const c10::intrusive_ptr<Raytracer> &self) { return self->gaussian_data; })
            .def("get_metadata", [](const c10::intrusive_ptr<Raytracer> &self) { return self->meta_data; })
            .def("get_stats", [](const c10::intrusive_ptr<Raytracer> &self) { return self->stats_data; })
            .def("get_ppll_forward_data",
                 [](const c10::intrusive_ptr<Raytracer> &self) { return self->ppll_forward_data; })
            .def("get_ppll_backward_data",
                 [](const c10::intrusive_ptr<Raytracer> &self) { return self->ppll_backward_data; })
            .def("get_num_channels", [](const c10::intrusive_ptr<Raytracer> &self) { return (int64_t)CHANNELS; })
            .def("get_max_alpha", [](const c10::intrusive_ptr<Raytracer> &self) { return (double)MAX_ALPHA; });
    }
};
