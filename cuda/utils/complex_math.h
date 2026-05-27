#pragma once

struct HarmonicComplex {
    float real;
    float imag;
};

__device__ __forceinline__ HarmonicComplex make_complex_value(float real = 0.0f, float imag = 0.0f) {
    return HarmonicComplex{real, imag};
}

__device__ __forceinline__ HarmonicComplex add_complex(HarmonicComplex lhs, HarmonicComplex rhs) {
    return make_complex_value(lhs.real + rhs.real, lhs.imag + rhs.imag);
}

__device__ __forceinline__ HarmonicComplex mul_complex(HarmonicComplex lhs, HarmonicComplex rhs) {
    return make_complex_value(lhs.real * rhs.real - lhs.imag * rhs.imag, lhs.real * rhs.imag + lhs.imag * rhs.real);
}

__device__ __forceinline__ HarmonicComplex scale_complex(HarmonicComplex value, float scale) {
    return make_complex_value(value.real * scale, value.imag * scale);
}

__device__ __forceinline__ HarmonicComplex mul_complex_i(HarmonicComplex value) {
    return make_complex_value(-value.imag, value.real);
}

__device__ __forceinline__ int complex_term_index(int degree, int order) {
    return degree * (degree + 1) / 2 + order;
}
