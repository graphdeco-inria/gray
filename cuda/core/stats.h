#pragma once

struct Stats {
    int *num_gaussians_hit;
    int *num_gaussians_accumulated;
};

#ifndef __CUDACC__
#include "headers.h"

struct StatsDataHolder : torch::CustomClassHolder {
    Tensor num_gaussians_hit;
    Tensor num_gaussians_accumulated;

    StatsDataHolder(uint32_t image_width, uint32_t image_height) {
        num_gaussians_hit = torch::zeros({(int64_t)image_height, (int64_t)image_width}, torch::kInt32).cuda();
        num_gaussians_accumulated = torch::zeros({(int64_t)image_height, (int64_t)image_width}, torch::kInt32).cuda();
    }

    Stats reify() {
        return Stats{
            .num_gaussians_hit = reinterpret_cast<int *>(num_gaussians_hit.data_ptr()),
            .num_gaussians_accumulated = reinterpret_cast<int *>(num_gaussians_accumulated.data_ptr()),
        };
    }

    void reset() {
        num_gaussians_hit.zero_();
        num_gaussians_accumulated.zero_();
    }

    static void bind(torch::Library &m) {
        m.class_<StatsDataHolder>("StatsDataHolder")
            .def_readonly("num_gaussians_hit", &StatsDataHolder::num_gaussians_hit)
            .def_readonly("num_gaussians_accumulated", &StatsDataHolder::num_gaussians_accumulated)
            .def("reset", &StatsDataHolder::reset);
    }
};

#endif