//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <vector_functions.h>
#include <vector_types.h>
#include <array>
#include <type_traits>

#if !defined(__CUDACC_RTC__)
#include <cmath>
#include <cstdlib>
#endif

/* scalar functions used in vector functions */
#ifndef M_PIf
#define M_PIf 3.14159265358979323846f
#endif
#ifndef M_PI_2f
#define M_PI_2f 1.57079632679489661923f
#endif
#ifndef M_1_PIf
#define M_1_PIf 0.318309886183790671538f
#endif

#if !defined(__CUDACC__)

__inline__ __device__ int max(int a, int b) { return a > b ? a : b; }

__inline__ __device__ int min(int a, int b) { return a < b ? a : b; }

__inline__ __device__ long long max(long long a, long long b) { return a > b ? a : b; }

__inline__ __device__ long long min(long long a, long long b) { return a < b ? a : b; }

__inline__ __device__ unsigned int max(unsigned int a, unsigned int b) { return a > b ? a : b; }

__inline__ __device__ unsigned int min(unsigned int a, unsigned int b) { return a < b ? a : b; }

__inline__ __device__ unsigned long long max(unsigned long long a, unsigned long long b) { return a > b ? a : b; }

__inline__ __device__ unsigned long long min(unsigned long long a, unsigned long long b) { return a < b ? a : b; }

/** lerp */
__inline__ __device__ float lerp(const float a, const float b, const float t) { return a + t * (b - a); }

/** bilerp */
__inline__ __device__ float bilerp(const float x00, const float x10, const float x01, const float x11, const float u,
                                   const float v) {
    return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

template <typename IntegerType> __inline__ __device__ IntegerType roundUp(IntegerType x, IntegerType y) {
    return ((x + y - 1) / y) * y;
}

#endif

/** clamp */
__inline__ __device__ float clamp(const float f, const float a, const float b) { return fmaxf(a, fminf(f, b)); }

/* float2 functions */
/******************************************************************************/

/** additional constructors
 * @{
 */
__inline__ __device__ float2 make_float2(const float s) { return make_float2(s, s); }
__inline__ __device__ float2 make_float2(const int2 &a) { return make_float2(float(a.x), float(a.y)); }
__inline__ __device__ float2 make_float2(const uint2 &a) { return make_float2(float(a.x), float(a.y)); }
/** @} */

/** negate */
__inline__ __device__ float2 operator-(const float2 &a) { return make_float2(-a.x, -a.y); }

/** min
 * @{
 */
__inline__ __device__ float2 fminf(const float2 &a, const float2 &b) {
    return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}
__inline__ __device__ float fminf(const float2 &a) { return fminf(a.x, a.y); }
/** @} */

/** max
 * @{
 */
__inline__ __device__ float2 fmaxf(const float2 &a, const float2 &b) {
    return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}
__inline__ __device__ float fmaxf(const float2 &a) { return fmaxf(a.x, a.y); }
/** @} */

/** add
 * @{
 */
__inline__ __device__ float2 operator+(const float2 &a, const float2 &b) { return make_float2(a.x + b.x, a.y + b.y); }
__inline__ __device__ float2 operator+(const float2 &a, const float b) { return make_float2(a.x + b, a.y + b); }
__inline__ __device__ float2 operator+(const float a, const float2 &b) { return make_float2(a + b.x, a + b.y); }
__inline__ __device__ void operator+=(float2 &a, const float2 &b) {
    a.x += b.x;
    a.y += b.y;
}
/** @} */

/** subtract
 * @{
 */
__inline__ __device__ float2 operator-(const float2 &a, const float2 &b) { return make_float2(a.x - b.x, a.y - b.y); }
__inline__ __device__ float2 operator-(const float2 &a, const float b) { return make_float2(a.x - b, a.y - b); }
__inline__ __device__ float2 operator-(const float a, const float2 &b) { return make_float2(a - b.x, a - b.y); }
__inline__ __device__ void operator-=(float2 &a, const float2 &b) {
    a.x -= b.x;
    a.y -= b.y;
}
/** @} */

/** multiply
 * @{
 */
__inline__ __device__ float2 operator*(const float2 &a, const float2 &b) { return make_float2(a.x * b.x, a.y * b.y); }
__inline__ __device__ float2 operator*(const float2 &a, const float s) { return make_float2(a.x * s, a.y * s); }
__inline__ __device__ float2 operator*(const float s, const float2 &a) { return make_float2(a.x * s, a.y * s); }
__inline__ __device__ void operator*=(float2 &a, const float2 &s) {
    a.x *= s.x;
    a.y *= s.y;
}
__inline__ __device__ void operator*=(float2 &a, const float s) {
    a.x *= s;
    a.y *= s;
}
/** @} */

/** divide
 * @{
 */
__inline__ __device__ float2 operator/(const float2 &a, const float2 &b) { return make_float2(a.x / b.x, a.y / b.y); }
__inline__ __device__ float2 operator/(const float2 &a, const float s) {
    float inv = 1.0f / s;
    return a * inv;
}
__inline__ __device__ float2 operator/(const float s, const float2 &a) { return make_float2(s / a.x, s / a.y); }
__inline__ __device__ void operator/=(float2 &a, const float s) {
    float inv = 1.0f / s;
    a *= inv;
}
/** @} */

/** lerp */
__inline__ __device__ float2 lerp(const float2 &a, const float2 &b, const float t) { return a + t * (b - a); }

/** bilerp */
__inline__ __device__ float2 bilerp(const float2 &x00, const float2 &x10, const float2 &x01, const float2 &x11,
                                    const float u, const float v) {
    return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp
 * @{
 */
__inline__ __device__ float2 clamp(const float2 &v, const float a, const float b) {
    return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

__inline__ __device__ float2 clamp(const float2 &v, const float2 &a, const float2 &b) {
    return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}
/** @} */

/** dot product */
__inline__ __device__ float dot(const float2 &a, const float2 &b) { return a.x * b.x + a.y * b.y; }

/** length */
__inline__ __device__ float length(const float2 &v) { return sqrtf(dot(v, v)); }

/** normalize */
__inline__ __device__ float2 normalize(const float2 &v) {
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

/** floor */
__inline__ __device__ float2 floor(const float2 &v) { return make_float2(::floorf(v.x), ::floorf(v.y)); }

/** reflect */
__inline__ __device__ float2 reflect(const float2 &i, const float2 &n) { return i - 2.0f * n * dot(n, i); }

/** Faceforward
 * Returns N if dot(i, nref) > 0; else -N;
 * Typical usage is N = faceforward(N, -ray.dir, N);
 * Note that this is opposite of what faceforward does in Cg and GLSL */
__inline__ __device__ float2 faceforward(const float2 &n, const float2 &i, const float2 &nref) {
    return n * copysignf(1.0f, dot(i, nref));
}

/** exp */
__inline__ __device__ float2 expf(const float2 &v) { return make_float2(::expf(v.x), ::expf(v.y)); }

/** If used on the device, this could place the the 'v' in local memory */
__inline__ __device__ float getByIndex(const float2 &v, int i) { return ((float *)(&v))[i]; }

/** If used on the device, this could place the the 'v' in local memory */
__inline__ __device__ void setByIndex(float2 &v, int i, float x) { ((float *)(&v))[i] = x; }

/* float3 functions */
/******************************************************************************/

/** additional constructors
 * @{
 */
__inline__ __device__ float3 make_float3(const float s) { return make_float3(s, s, s); }
__inline__ __device__ float3 make_float3(const float2 &a) { return make_float3(a.x, a.y, 0.0f); }
__inline__ __device__ float3 make_float3(const int3 &a) { return make_float3(float(a.x), float(a.y), float(a.z)); }
__inline__ __device__ float3 make_float3(const uint3 &a) { return make_float3(float(a.x), float(a.y), float(a.z)); }
/** @} */

/** negate */
__inline__ __device__ float3 operator-(const float3 &a) { return make_float3(-a.x, -a.y, -a.z); }

/** min
 * @{
 */
__inline__ __device__ float3 fminf(const float3 &a, const float3 &b) {
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}
__inline__ __device__ float fminf(const float3 &a) { return fminf(fminf(a.x, a.y), a.z); }
/** @} */

/** max
 * @{
 */
__inline__ __device__ float3 fmaxf(const float3 &a, const float3 &b) {
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}
__inline__ __device__ float fmaxf(const float3 &a) { return fmaxf(fmaxf(a.x, a.y), a.z); }
/** @} */

/** add
 * @{
 */
__inline__ __device__ float3 operator+(const float3 &a, const float3 &b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__inline__ __device__ float3 operator+(const float3 &a, const float b) {
    return make_float3(a.x + b, a.y + b, a.z + b);
}
__inline__ __device__ float3 operator+(const float a, const float3 &b) {
    return make_float3(a + b.x, a + b.y, a + b.z);
}
__inline__ __device__ void operator+=(float3 &a, const float3 &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}
/** @} */

/** subtract
 * @{
 */
__inline__ __device__ float3 operator-(const float3 &a, const float3 &b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__inline__ __device__ float3 operator-(const float3 &a, const float b) {
    return make_float3(a.x - b, a.y - b, a.z - b);
}
__inline__ __device__ float3 operator-(const float a, const float3 &b) {
    return make_float3(a - b.x, a - b.y, a - b.z);
}
__inline__ __device__ void operator-=(float3 &a, const float3 &b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}
/** @} */

/** multiply
 * @{
 */
__inline__ __device__ float3 operator*(const float3 &a, const float3 &b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
__inline__ __device__ float3 operator*(const float3 &a, const float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__inline__ __device__ float3 operator*(const float s, const float3 &a) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__inline__ __device__ void operator*=(float3 &a, const float3 &s) {
    a.x *= s.x;
    a.y *= s.y;
    a.z *= s.z;
}
__inline__ __device__ void operator*=(float3 &a, const float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
}
/** @} */

/** divide
 * @{
 */
__inline__ __device__ float3 operator/(const float3 &a, const float3 &b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
__inline__ __device__ float3 operator/(const float3 &a, const float s) {
    float inv = 1.0f / s;
    return a * inv;
}
__inline__ __device__ float3 operator/(const float s, const float3 &a) {
    return make_float3(s / a.x, s / a.y, s / a.z);
}
__inline__ __device__ void operator/=(float3 &a, const float s) {
    float inv = 1.0f / s;
    a *= inv;
}
/** @} */

/** lerp */
__inline__ __device__ float3 lerp(const float3 &a, const float3 &b, const float t) { return a + t * (b - a); }

/** bilerp */
__inline__ __device__ float3 bilerp(const float3 &x00, const float3 &x10, const float3 &x01, const float3 &x11,
                                    const float u, const float v) {
    return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp
 * @{
 */
__inline__ __device__ float3 clamp(const float3 &v, const float a, const float b) {
    return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

__inline__ __device__ float3 clamp(const float3 &v, const float3 &a, const float3 &b) {
    return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}
/** @} */

/** dot product */
__inline__ __device__ float dot(const float3 &a, const float3 &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

/** cross product */
__inline__ __device__ float3 cross(const float3 &a, const float3 &b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

/** length */
__inline__ __device__ float length(const float3 &v) { return sqrtf(dot(v, v)); }

/** normalize */
__inline__ __device__ float3 normalize(const float3 &v) {
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

/** floor */
__inline__ __device__ float3 floor(const float3 &v) { return make_float3(::floorf(v.x), ::floorf(v.y), ::floorf(v.z)); }

/** reflect */
__inline__ __device__ float3 reflect(const float3 &i, const float3 &n) { return i - 2.0f * n * dot(n, i); }

/** Faceforward
 * Returns N if dot(i, nref) > 0; else -N;
 * Typical usage is N = faceforward(N, -ray.dir, N);
 * Note that this is opposite of what faceforward does in Cg and GLSL */
__inline__ __device__ float3 faceforward(const float3 &n, const float3 &i, const float3 &nref) {
    return n * copysignf(1.0f, dot(i, nref));
}

/** exp */
__inline__ __device__ float3 expf(const float3 &v) { return make_float3(::expf(v.x), ::expf(v.y), ::expf(v.z)); }

/** If used on the device, this could place the the 'v' in local memory */
__inline__ __device__ float getByIndex(const float3 &v, int i) { return ((float *)(&v))[i]; }

/** If used on the device, this could place the the 'v' in local memory */
__inline__ __device__ void setByIndex(float3 &v, int i, float x) { ((float *)(&v))[i] = x; }

/* float4 functions */
/******************************************************************************/

/** additional constructors
 * @{
 */
__inline__ __device__ float4 make_float4(const float s) { return make_float4(s, s, s, s); }
__inline__ __device__ float4 make_float4(const float3 &a) { return make_float4(a.x, a.y, a.z, 0.0f); }
__inline__ __device__ float4 make_float4(const int4 &a) {
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
__inline__ __device__ float4 make_float4(const uint4 &a) {
    return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}
/** @} */

/** negate */
__inline__ __device__ float4 operator-(const float4 &a) { return make_float4(-a.x, -a.y, -a.z, -a.w); }

/** min
 * @{
 */
__inline__ __device__ float4 fminf(const float4 &a, const float4 &b) {
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}
__inline__ __device__ float fminf(const float4 &a) { return fminf(fminf(a.x, a.y), fminf(a.z, a.w)); }
/** @} */

/** max
 * @{
 */
__inline__ __device__ float4 fmaxf(const float4 &a, const float4 &b) {
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}
__inline__ __device__ float fmaxf(const float4 &a) { return fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)); }
/** @} */

/** add
 * @{
 */
__inline__ __device__ float4 operator+(const float4 &a, const float4 &b) {
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__inline__ __device__ float4 operator+(const float4 &a, const float b) {
    return make_float4(a.x + b, a.y + b, a.z + b, a.w + b);
}
__inline__ __device__ float4 operator+(const float a, const float4 &b) {
    return make_float4(a + b.x, a + b.y, a + b.z, a + b.w);
}
__inline__ __device__ void operator+=(float4 &a, const float4 &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
/** @} */

/** subtract
 * @{
 */
__inline__ __device__ float4 operator-(const float4 &a, const float4 &b) {
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__inline__ __device__ float4 operator-(const float4 &a, const float b) {
    return make_float4(a.x - b, a.y - b, a.z - b, a.w - b);
}
__inline__ __device__ float4 operator-(const float a, const float4 &b) {
    return make_float4(a - b.x, a - b.y, a - b.z, a - b.w);
}
__inline__ __device__ void operator-=(float4 &a, const float4 &b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
/** @} */

/** multiply
 * @{
 */
__inline__ __device__ float4 operator*(const float4 &a, const float4 &s) {
    return make_float4(a.x * s.x, a.y * s.y, a.z * s.z, a.w * s.w);
}
__inline__ __device__ float4 operator*(const float4 &a, const float s) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__inline__ __device__ float4 operator*(const float s, const float4 &a) {
    return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__inline__ __device__ void operator*=(float4 &a, const float4 &s) {
    a.x *= s.x;
    a.y *= s.y;
    a.z *= s.z;
    a.w *= s.w;
}
__inline__ __device__ void operator*=(float4 &a, const float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    a.w *= s;
}
/** @} */

/** divide
 * @{
 */
__inline__ __device__ float4 operator/(const float4 &a, const float4 &b) {
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
__inline__ __device__ float4 operator/(const float4 &a, const float s) {
    float inv = 1.0f / s;
    return a * inv;
}
__inline__ __device__ float4 operator/(const float s, const float4 &a) {
    return make_float4(s / a.x, s / a.y, s / a.z, s / a.w);
}
__inline__ __device__ void operator/=(float4 &a, const float s) {
    float inv = 1.0f / s;
    a *= inv;
}
/** @} */

/** lerp */
__inline__ __device__ float4 lerp(const float4 &a, const float4 &b, const float t) { return a + t * (b - a); }

/** bilerp */
__inline__ __device__ float4 bilerp(const float4 &x00, const float4 &x10, const float4 &x01, const float4 &x11,
                                    const float u, const float v) {
    return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

/** clamp
 * @{
 */
__inline__ __device__ float4 clamp(const float4 &v, const float a, const float b) {
    return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

__inline__ __device__ float4 clamp(const float4 &v, const float4 &a, const float4 &b) {
    return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}
/** @} */

/** dot product */
__inline__ __device__ float dot(const float4 &a, const float4 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/** length */
__inline__ __device__ float length(const float4 &r) { return sqrtf(dot(r, r)); }

/** normalize */
__inline__ __device__ float4 normalize(const float4 &v) {
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

/** floor */
__inline__ __device__ float4 floor(const float4 &v) {
    return make_float4(::floorf(v.x), ::floorf(v.y), ::floorf(v.z), ::floorf(v.w));
}

/** reflect */
__inline__ __device__ float4 reflect(const float4 &i, const float4 &n) { return i - 2.0f * n * dot(n, i); }

/**
 * Faceforward
 * Returns N if dot(i, nref) > 0; else -N;
 * Typical usage is N = faceforward(N, -ray.dir, N);
 * Note that this is opposite of what faceforward does in Cg and GLSL
 */
__inline__ __device__ float4 faceforward(const float4 &n, const float4 &i, const float4 &nref) {
    return n * copysignf(1.0f, dot(i, nref));
}

/** exp */
__inline__ __device__ float4 expf(const float4 &v) {
    return make_float4(::expf(v.x), ::expf(v.y), ::expf(v.z), ::expf(v.w));
}

/** If used on the device, this could place the the 'v' in local memory */
__inline__ __device__ float getByIndex(const float4 &v, int i) { return ((float *)(&v))[i]; }

/** If used on the device, this could place the the 'v' in local memory */
__inline__ __device__ void setByIndex(float4 &v, int i, float x) { ((float *)(&v))[i] = x; }

/******************************************************************************/

/** Narrowing functions
 * @{
 */
__inline__ __device__ int2 make_int2(const int3 &v0) { return make_int2(v0.x, v0.y); }
__inline__ __device__ int2 make_int2(const int4 &v0) { return make_int2(v0.x, v0.y); }
__inline__ __device__ int3 make_int3(const int4 &v0) { return make_int3(v0.x, v0.y, v0.z); }
__inline__ __device__ uint2 make_uint2(const uint3 &v0) { return make_uint2(v0.x, v0.y); }
__inline__ __device__ uint2 make_uint2(const uint4 &v0) { return make_uint2(v0.x, v0.y); }
__inline__ __device__ uint3 make_uint3(const uint4 &v0) { return make_uint3(v0.x, v0.y, v0.z); }
__inline__ __device__ longlong2 make_longlong2(const longlong3 &v0) { return make_longlong2(v0.x, v0.y); }
__inline__ __device__ longlong2 make_longlong2(const longlong4 &v0) { return make_longlong2(v0.x, v0.y); }
__inline__ __device__ longlong3 make_longlong3(const longlong4 &v0) { return make_longlong3(v0.x, v0.y, v0.z); }
__inline__ __device__ ulonglong2 make_ulonglong2(const ulonglong3 &v0) { return make_ulonglong2(v0.x, v0.y); }
__inline__ __device__ ulonglong2 make_ulonglong2(const ulonglong4 &v0) { return make_ulonglong2(v0.x, v0.y); }
__inline__ __device__ ulonglong3 make_ulonglong3(const ulonglong4 &v0) { return make_ulonglong3(v0.x, v0.y, v0.z); }
__inline__ __device__ float2 make_float2(const float3 &v0) { return make_float2(v0.x, v0.y); }
__inline__ __device__ float2 make_float2(const float4 &v0) { return make_float2(v0.x, v0.y); }
__inline__ __device__ float3 make_float3(const float4 &v0) { return make_float3(v0.x, v0.y, v0.z); }
/** @} */

/** Assemble functions from smaller vectors
 * @{
 */
__inline__ __device__ int3 make_int3(const int v0, const int2 &v1) { return make_int3(v0, v1.x, v1.y); }
__inline__ __device__ int3 make_int3(const int2 &v0, const int v1) { return make_int3(v0.x, v0.y, v1); }
__inline__ __device__ int4 make_int4(const int v0, const int v1, const int2 &v2) {
    return make_int4(v0, v1, v2.x, v2.y);
}
__inline__ __device__ int4 make_int4(const int v0, const int2 &v1, const int v2) {
    return make_int4(v0, v1.x, v1.y, v2);
}
__inline__ __device__ int4 make_int4(const int2 &v0, const int v1, const int v2) {
    return make_int4(v0.x, v0.y, v1, v2);
}
__inline__ __device__ int4 make_int4(const int v0, const int3 &v1) { return make_int4(v0, v1.x, v1.y, v1.z); }
__inline__ __device__ int4 make_int4(const int3 &v0, const int v1) { return make_int4(v0.x, v0.y, v0.z, v1); }
__inline__ __device__ int4 make_int4(const int2 &v0, const int2 &v1) { return make_int4(v0.x, v0.y, v1.x, v1.y); }
__inline__ __device__ uint3 make_uint3(const unsigned int v0, const uint2 &v1) { return make_uint3(v0, v1.x, v1.y); }
__inline__ __device__ uint3 make_uint3(const uint2 &v0, const unsigned int v1) { return make_uint3(v0.x, v0.y, v1); }
__inline__ __device__ uint4 make_uint4(const unsigned int v0, const unsigned int v1, const uint2 &v2) {
    return make_uint4(v0, v1, v2.x, v2.y);
}
__inline__ __device__ uint4 make_uint4(const unsigned int v0, const uint2 &v1, const unsigned int v2) {
    return make_uint4(v0, v1.x, v1.y, v2);
}
__inline__ __device__ uint4 make_uint4(const uint2 &v0, const unsigned int v1, const unsigned int v2) {
    return make_uint4(v0.x, v0.y, v1, v2);
}
__inline__ __device__ uint4 make_uint4(const unsigned int v0, const uint3 &v1) {
    return make_uint4(v0, v1.x, v1.y, v1.z);
}
__inline__ __device__ uint4 make_uint4(const uint3 &v0, const unsigned int v1) {
    return make_uint4(v0.x, v0.y, v0.z, v1);
}
__inline__ __device__ uint4 make_uint4(const uint2 &v0, const uint2 &v1) { return make_uint4(v0.x, v0.y, v1.x, v1.y); }
__inline__ __device__ longlong3 make_longlong3(const long long v0, const longlong2 &v1) {
    return make_longlong3(v0, v1.x, v1.y);
}
__inline__ __device__ longlong3 make_longlong3(const longlong2 &v0, const long long v1) {
    return make_longlong3(v0.x, v0.y, v1);
}
__inline__ __device__ longlong4 make_longlong4(const long long v0, const long long v1, const longlong2 &v2) {
    return make_longlong4(v0, v1, v2.x, v2.y);
}
__inline__ __device__ longlong4 make_longlong4(const long long v0, const longlong2 &v1, const long long v2) {
    return make_longlong4(v0, v1.x, v1.y, v2);
}
__inline__ __device__ longlong4 make_longlong4(const longlong2 &v0, const long long v1, const long long v2) {
    return make_longlong4(v0.x, v0.y, v1, v2);
}
__inline__ __device__ longlong4 make_longlong4(const long long v0, const longlong3 &v1) {
    return make_longlong4(v0, v1.x, v1.y, v1.z);
}
__inline__ __device__ longlong4 make_longlong4(const longlong3 &v0, const long long v1) {
    return make_longlong4(v0.x, v0.y, v0.z, v1);
}
__inline__ __device__ longlong4 make_longlong4(const longlong2 &v0, const longlong2 &v1) {
    return make_longlong4(v0.x, v0.y, v1.x, v1.y);
}
__inline__ __device__ ulonglong3 make_ulonglong3(const unsigned long long v0, const ulonglong2 &v1) {
    return make_ulonglong3(v0, v1.x, v1.y);
}
__inline__ __device__ ulonglong3 make_ulonglong3(const ulonglong2 &v0, const unsigned long long v1) {
    return make_ulonglong3(v0.x, v0.y, v1);
}
__inline__ __device__ ulonglong4 make_ulonglong4(const unsigned long long v0, const unsigned long long v1,
                                                 const ulonglong2 &v2) {
    return make_ulonglong4(v0, v1, v2.x, v2.y);
}
__inline__ __device__ ulonglong4 make_ulonglong4(const unsigned long long v0, const ulonglong2 &v1,
                                                 const unsigned long long v2) {
    return make_ulonglong4(v0, v1.x, v1.y, v2);
}
__inline__ __device__ ulonglong4 make_ulonglong4(const ulonglong2 &v0, const unsigned long long v1,
                                                 const unsigned long long v2) {
    return make_ulonglong4(v0.x, v0.y, v1, v2);
}
__inline__ __device__ ulonglong4 make_ulonglong4(const unsigned long long v0, const ulonglong3 &v1) {
    return make_ulonglong4(v0, v1.x, v1.y, v1.z);
}
__inline__ __device__ ulonglong4 make_ulonglong4(const ulonglong3 &v0, const unsigned long long v1) {
    return make_ulonglong4(v0.x, v0.y, v0.z, v1);
}
__inline__ __device__ ulonglong4 make_ulonglong4(const ulonglong2 &v0, const ulonglong2 &v1) {
    return make_ulonglong4(v0.x, v0.y, v1.x, v1.y);
}
__inline__ __device__ float3 make_float3(const float2 &v0, const float v1) { return make_float3(v0.x, v0.y, v1); }
__inline__ __device__ float3 make_float3(const float v0, const float2 &v1) { return make_float3(v0, v1.x, v1.y); }
__inline__ __device__ float4 make_float4(const float v0, const float v1, const float2 &v2) {
    return make_float4(v0, v1, v2.x, v2.y);
}
__inline__ __device__ float4 make_float4(const float v0, const float2 &v1, const float v2) {
    return make_float4(v0, v1.x, v1.y, v2);
}
__inline__ __device__ float4 make_float4(const float2 &v0, const float v1, const float v2) {
    return make_float4(v0.x, v0.y, v1, v2);
}
__inline__ __device__ float4 make_float4(const float v0, const float3 &v1) { return make_float4(v0, v1.x, v1.y, v1.z); }
__inline__ __device__ float4 make_float4(const float3 &v0, const float v1) { return make_float4(v0.x, v0.y, v0.z, v1); }
__inline__ __device__ float4 make_float4(const float2 &v0, const float2 &v1) {
    return make_float4(v0.x, v0.y, v1.x, v1.y);
}
/** @} */

/******************************************************************************/

//** Added functions

__inline__ __device__ float2 sqrtf(const float2 &v) { return make_float2(::sqrtf(v.x), ::sqrtf(v.y)); }

__inline__ __device__ float3 sqrtf(const float3 &v) { return make_float3(::sqrtf(v.x), ::sqrtf(v.y), ::sqrtf(v.z)); }

__inline__ __device__ float4 sqrtf(const float4 &v) {
    return make_float4(::sqrtf(v.x), ::sqrtf(v.y), ::sqrtf(v.z), ::sqrtf(v.w));
}

// Generic templated vector type and math functions

// generic_float<K> definition

template <size_t K> struct generic_float {
    float v[K];
    __inline__ __device__ generic_float() {
        for (int i = 0; i < K; ++i)
            v[i] = 0.0f;
    }
    __inline__ __device__ explicit generic_float(float val) {
        for (int i = 0; i < K; ++i)
            v[i] = val;
    }
    template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == K>>
    __inline__ __device__ generic_float(Args... args) : v{static_cast<float>(args)...} {}
    __inline__ __device__ float &operator[](size_t i) { return v[i]; }
    __inline__ __device__ const float &operator[](size_t i) const { return v[i]; }
};

// make_generic_float

template <size_t K> __inline__ __device__ generic_float<K> make_generic_float(const float s) {
    return generic_float<K>(s);
}

template <size_t K> __inline__ __device__ generic_float<K> make_generic_float(const float *arr) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = arr[i];
    return r;
}

// Negate
template <size_t K> __inline__ __device__ generic_float<K> operator-(const generic_float<K> &a) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = -a[i];
    return r;
}

// Min/max
template <size_t K> __inline__ __device__ generic_float<K> fminf(const generic_float<K> &a, const generic_float<K> &b) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = fminf(a[i], b[i]);
    return r;
}
template <size_t K> __inline__ __device__ float fminf(const generic_float<K> &a) {
    float m = a[0];
    for (size_t i = 1; i < K; ++i)
        m = fminf(m, a[i]);
    return m;
}
template <size_t K> __inline__ __device__ generic_float<K> fmaxf(const generic_float<K> &a, const generic_float<K> &b) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = fmaxf(a[i], b[i]);
    return r;
}
template <size_t K> __inline__ __device__ float fmaxf(const generic_float<K> &a) {
    float m = a[0];
    for (size_t i = 1; i < K; ++i)
        m = fmaxf(m, a[i]);
    return m;
}

// Add
template <size_t K>
__inline__ __device__ generic_float<K> operator+(const generic_float<K> &a, const generic_float<K> &b) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = a[i] + b[i];
    return r;
}
template <size_t K> __inline__ __device__ generic_float<K> operator+(const generic_float<K> &a, float b) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = a[i] + b;
    return r;
}
template <size_t K> __inline__ __device__ generic_float<K> operator+(float a, const generic_float<K> &b) {
    return b + a;
}
template <size_t K> __inline__ __device__ void operator+=(generic_float<K> &a, const generic_float<K> &b) {
    for (size_t i = 0; i < K; ++i)
        a[i] += b[i];
}

// Subtract
template <size_t K>
__inline__ __device__ generic_float<K> operator-(const generic_float<K> &a, const generic_float<K> &b) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = a[i] - b[i];
    return r;
}
template <size_t K> __inline__ __device__ generic_float<K> operator-(const generic_float<K> &a, float b) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = a[i] - b;
    return r;
}
template <size_t K> __inline__ __device__ generic_float<K> operator-(float a, const generic_float<K> &b) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = a - b[i];
    return r;
}
template <size_t K> __inline__ __device__ void operator-=(generic_float<K> &a, const generic_float<K> &b) {
    for (size_t i = 0; i < K; ++i)
        a[i] -= b[i];
}

// Multiply
template <size_t K>
__inline__ __device__ generic_float<K> operator*(const generic_float<K> &a, const generic_float<K> &b) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = a[i] * b[i];
    return r;
}
template <size_t K> __inline__ __device__ generic_float<K> operator*(const generic_float<K> &a, float s) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = a[i] * s;
    return r;
}
template <size_t K> __inline__ __device__ generic_float<K> operator*(float s, const generic_float<K> &a) {
    return a * s;
}
template <size_t K> __inline__ __device__ void operator*=(generic_float<K> &a, const generic_float<K> &s) {
    for (size_t i = 0; i < K; ++i)
        a[i] *= s[i];
}
template <size_t K> __inline__ __device__ void operator*=(generic_float<K> &a, float s) {
    for (size_t i = 0; i < K; ++i)
        a[i] *= s;
}

// Divide
template <size_t K>
__inline__ __device__ generic_float<K> operator/(const generic_float<K> &a, const generic_float<K> &b) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = a[i] / b[i];
    return r;
}
template <size_t K> __inline__ __device__ generic_float<K> operator/(const generic_float<K> &a, float s) {
    float inv = 1.0f / s;
    return a * inv;
}
template <size_t K> __inline__ __device__ generic_float<K> operator/(float s, const generic_float<K> &a) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = s / a[i];
    return r;
}
template <size_t K> __inline__ __device__ void operator/=(generic_float<K> &a, float s) {
    float inv = 1.0f / s;
    a *= inv;
}

// lerp

template <size_t K>
__inline__ __device__ generic_float<K> lerp(const generic_float<K> &a, const generic_float<K> &b, float t) {
    return a + t * (b - a);
}

template <size_t K>
__inline__ __device__ generic_float<K> bilerp(const generic_float<K> &x00, const generic_float<K> &x10,
                                              const generic_float<K> &x01, const generic_float<K> &x11, float u,
                                              float v) {
    return lerp(lerp(x00, x10, u), lerp(x01, x11, u), v);
}

// clamp

template <size_t K> __inline__ __device__ generic_float<K> clamp(const generic_float<K> &v, float a, float b) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = clamp(v[i], a, b);
    return r;
}
template <size_t K>
__inline__ __device__ generic_float<K> clamp(const generic_float<K> &v, const generic_float<K> &a,
                                             const generic_float<K> &b) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = clamp(v[i], a[i], b[i]);
    return r;
}

// dot

template <size_t K> __inline__ __device__ float dot(const generic_float<K> &a, const generic_float<K> &b) {
    float r = 0.0f;
    for (size_t i = 0; i < K; ++i)
        r += a[i] * b[i];
    return r;
}

// length

template <size_t K> __inline__ __device__ float length(const generic_float<K> &v) { return sqrtf(dot(v, v)); }

// normalize

template <size_t K> __inline__ __device__ generic_float<K> normalize(const generic_float<K> &v) {
    float invLen = 1.0f / sqrtf(dot(v, v));
    return v * invLen;
}

// floor

template <size_t K> __inline__ __device__ generic_float<K> floor(const generic_float<K> &v) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = ::floorf(v[i]);
    return r;
}

// reflect

template <size_t K>
__inline__ __device__ generic_float<K> reflect(const generic_float<K> &i, const generic_float<K> &n) {
    return i - 2.0f * n * dot(n, i);
}

// faceforward

template <size_t K>
__inline__ __device__ generic_float<K> faceforward(const generic_float<K> &n, const generic_float<K> &i,
                                                   const generic_float<K> &nref) {
    return n * copysignf(1.0f, dot(i, nref));
}

// expf

template <size_t K> __inline__ __device__ generic_float<K> expf(const generic_float<K> &v) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = ::expf(v[i]);
    return r;
}

// getByIndex/setByIndex

template <size_t K> __inline__ __device__ float getByIndex(const generic_float<K> &v, int i) { return v[i]; }
template <size_t K> __inline__ __device__ void setByIndex(generic_float<K> &v, int i, float x) { v[i] = x; }

// sqrtf

template <size_t K> __inline__ __device__ generic_float<K> sqrtf(const generic_float<K> &v) {
    generic_float<K> r;
    for (size_t i = 0; i < K; ++i)
        r[i] = ::sqrtf(v[i]);
    return r;
}

// ----------------------

#if CHANNELS == 3
#define floatK float3
#define make_floatK(x) make_float3(x)
#else
#define floatK generic_float<CHANNELS>
#define make_floatK(x) make_generic_float<CHANNELS>(x)
#endif