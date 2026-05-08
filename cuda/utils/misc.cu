#pragma once

#include "cuda_fp16.h"
__device__ __inline__ void atomicAddX(float *address, float val) { atomicAdd(address, val); }

__device__ __inline__ void atomicAddX(float2 *address, float2 val) {
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
}

__device__ __inline__ void atomicAddX(float3 *address, float3 val) {
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
    atomicAdd(&address->z, val.z);
}

__device__ __inline__ void atomicAddX(float4 *address, float4 val) {
    atomicAdd(&address->x, val.x);
    atomicAdd(&address->y, val.y);
    atomicAdd(&address->z, val.z);
    atomicAdd(&address->w, val.w);
}

template <size_t K> __device__ __inline__ void atomicAddX(generic_float<K> *address, generic_float<K> val) {
#pragma unroll
    for (int i = 0; i < K; ++i)
        atomicAdd(&((*address)[i]), val[i]);
}

__device__ void fill_array(float *arr, uint32_t size, float val) {
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}

__device__ void fill_array(float2 *arr, uint32_t size, float2 val) {
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}

__device__ void fill_array(float3 *arr, uint32_t size, float3 val) {
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}

__device__ void fill_array(float4 *arr, uint32_t size, float4 val) {
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}

template <typename T> __device__ void fill_array(T *arr, uint32_t size, T val) {
    for (int i = 0; i < size; i++) {
        arr[i] = val;
    }
}

__device__ float dot(float a, float b) {
    return a * b; // * convenience function for generated code
}

__device__ unsigned int __forceinline__ packFloats(const float &distance, const float &alpha) {
    __half2 packed_halves = __halves2half2(__float2half(distance), __float2half(alpha));
    return *reinterpret_cast<const unsigned int *>(&packed_halves);
}

__device__ float __forceinline__ unpackDistance(const unsigned int &packed_bits) {
    __half2 packed_halves = *reinterpret_cast<const __half2 *>(&packed_bits);
    return __half2float(packed_halves.x);
}

__device__ float __forceinline__ unpackAlpha(const unsigned int &packed_bits) {
    __half2 packed_halves = *reinterpret_cast<const __half2 *>(&packed_bits);
    return __half2float(packed_halves.y);
}

__inline__ __device__ float sign(float v) {
    if (v == 0.0f)
        return 0.0f;
    return v < 0 ? -1.0f : 1.0f;
}

__inline__ __device__ float2 sign(float2 v) { return make_float2(sign(v.x), sign(v.y)); }

__inline__ __device__ float3 sign(float3 v) { return make_float3(sign(v.x), sign(v.y), sign(v.z)); }

__inline__ __device__ float4 sign(float4 v) { return make_float4(sign(v.x), sign(v.y), sign(v.z), sign(v.w)); }

template <size_t K> __inline__ __device__ generic_float<K> sign(generic_float<K> v) {
    generic_float<K> r;
#pragma unroll
    for (int i = 0; i < K; ++i)
        r[i] = copysignf(1.0f, v[i]);
    return r;
}
