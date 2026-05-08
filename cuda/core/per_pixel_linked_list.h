#pragma once

struct PerPixelLinkedList {
    static constexpr uint32_t NULL_PTR = std::numeric_limits<uint32_t>::max();

    char name[9];
    uint32_t *head_per_pixel;
    uint32_t *total_hits;

    uint32_t *gaussian_ids;
    float *distances;
    float *alphas;

    uint32_t *previous_entries;

    uint32_t size;

#ifdef __CUDACC__
    __device__ void insert(bool grads_enabled, uint32_t pixel_id, uint32_t gaussian_id, float distance, float alpha) {
        int hit_idx = atomicAdd(total_hits, 1);
        if (hit_idx >= size) {
            printf("Fatal: %s PPLL overflow! Increase the size of the %s per-pixel linked list.\n", name, name);
            __trap();
        }
        gaussian_ids[hit_idx] = gaussian_id;
        distances[hit_idx] = distance;
        alphas[hit_idx] = alpha;
        previous_entries[hit_idx] = head_per_pixel[pixel_id];
        head_per_pixel[pixel_id] = hit_idx;
    }

    __device__ void reset(uint32_t pixel_id) { head_per_pixel[pixel_id] = NULL_PTR; }

    struct PixelIterator {
        const PerPixelLinkedList *parent;
        uint32_t hit_idx;

        __device__ uint32_t operator*() const { return hit_idx; }

        __device__ PixelIterator &operator++() {
            hit_idx = parent->previous_entries[hit_idx];
            return *this;
        }

        __device__ bool operator!=(const PixelIterator &other) const { return hit_idx != other.hit_idx; }
    };

    struct PixelView {
        const PerPixelLinkedList *parent;
        uint32_t pixel_id;

        __device__ PixelIterator begin() const { return PixelIterator{parent, parent->head_per_pixel[pixel_id]}; }

        __device__ PixelIterator end() const { return PixelIterator{parent, PerPixelLinkedList::NULL_PTR}; }
    };

    __device__ PixelView pixel_view(uint32_t pixel_id) const { return PixelView{this, pixel_id}; }
#endif
};

#ifndef __CUDACC__
#include "headers.h"

struct PPLLDataHolder : torch::CustomClassHolder {
    char name[9];
    Tensor head_per_pixel;
    Tensor total_hits = torch::zeros({1}, CUDA_INT32) - 1;
    Tensor gaussian_ids;
    Tensor distances;
    Tensor alphas;
    Tensor previous_entries;
    uint32_t size;

    PPLLDataHolder(const std::string &name_, uint32_t image_width, uint32_t image_height, uint32_t size_) {
        strncpy(name, name_.c_str(), 8);
        name[8] = '\0';
        head_per_pixel = torch::zeros({image_height, image_width}, CUDA_INT32);
        head_per_pixel.fill_((int)PerPixelLinkedList::NULL_PTR);
        gaussian_ids = torch::zeros({size_}, CUDA_INT32);
        distances = torch::zeros({size_}, CUDA_FLOAT32);
        alphas = torch::zeros({size_}, CUDA_FLOAT32);
        previous_entries = torch::zeros({size_}, CUDA_INT32);
        size = size_;
    }

    PerPixelLinkedList reify() {
        PerPixelLinkedList pll = {.head_per_pixel = reinterpret_cast<uint32_t *>(head_per_pixel.data_ptr()),
                                  .total_hits = reinterpret_cast<uint32_t *>(total_hits.data_ptr()),
                                  .gaussian_ids = reinterpret_cast<uint32_t *>(gaussian_ids.data_ptr()),
                                  .distances = reinterpret_cast<float *>(distances.data_ptr()),
                                  .alphas = reinterpret_cast<float *>(alphas.data_ptr()),
                                  .previous_entries = reinterpret_cast<uint32_t *>(previous_entries.data_ptr()),
                                  .size = size};
        strncpy(pll.name, name, 8);
        return pll;
    }

    void reset() {
        total_hits.fill_(0);
        head_per_pixel.fill_((int)PerPixelLinkedList::NULL_PTR);
    }

    static void bind(torch::Library &m) {
        m.class_<PPLLDataHolder>("PPLLDataHolder")
            .def_readonly("head_per_pixel", &PPLLDataHolder::head_per_pixel)
            .def_readonly("total_hits", &PPLLDataHolder::total_hits)
            .def_readonly("gaussian_ids", &PPLLDataHolder::gaussian_ids)
            .def_readonly("distances", &PPLLDataHolder::distances)
            .def_readonly("alphas", &PPLLDataHolder::alphas)
            .def_readonly("previous_entries", &PPLLDataHolder::previous_entries)

            .def_static("NULL_PTR", []() { return (int64_t)PerPixelLinkedList::NULL_PTR; });
    }
};
#endif