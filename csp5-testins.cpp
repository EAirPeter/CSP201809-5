#pragma GCC target("avx2")

#include <immintrin.h>

#include <chrono>
#include <iostream>
#include <random>

using namespace std;
using namespace chrono;

constexpr uint32_t Mod = 998244353;
constexpr uint64_t ModMod = 4287426850549923841;
constexpr uint32_t BmM = 288737297;
constexpr uint32_t BmW = 29;

constexpr uint32_t MAdd(uint32_t A, uint32_t B) {
    return A + B < Mod ? A + B : A + B - Mod;
}

constexpr uint32_t MSub(uint32_t A, uint32_t B) {
    return A < B ? A - B + Mod : A - B;
}

constexpr uint32_t MMul(uint32_t A, uint32_t B) {
    return (uint32_t) ((uint64_t) A * B % Mod);
}

const __m256i vzero __attribute__((aligned(32))) = {0, 0, 0, 0};
const __m256i vm32 __attribute__((aligned(32))) = {ModMod, ModMod, ModMod, ModMod};
const __m256i vm64 __attribute__((aligned(32))) = {Mod, Mod, Mod, Mod};
const __m256i vbmm __attribute__((aligned(32))) = {BmM, BmM, BmM, BmM};
const __m256i vm0 __attribute__((aligned(32))) = {
    0x1111111103020100,
    0x111111110b0a0908,
    0x1111111113121110,
    0x111111111b1a1918
};
const __m256i vm1 __attribute__((aligned(32))) = {
    0x1111111107060504,
    0x111111110f0e0d0c,
    0x1111111117161514,
    0x111111111f1e1d1c
};

inline __m256i VEx0(__m256i v) {
    return _mm256_shuffle_epi8(v, vm0);
}

inline __m256i VEx1(__m256i v) {
    return _mm256_shuffle_epi8(v, vm1);
}

inline __m256i VIntlv(__m256i v0, __m256i v1) {
    return _mm256_blend_epi32(v0, _mm256_shuffle_epi32(v1, 0xb1), 0xaa);
}

inline __m256i VAdd(__m256i va, __m256i vb) {
    __m256i vra = _mm256_add_epi32(va, vb);
    __m256i vrb = _mm256_sub_epi32(vra, vm32);
    return _mm256_min_epu32(vra, vrb);
}

inline __m256i VSub(__m256i va, __m256i vb) {
    __m256i vra = _mm256_sub_epi32(va, vb);
    __m256i vrb = _mm256_add_epi32(vra, vm32);
    return _mm256_min_epu32(vra, vrb);
}

inline __m256i VMul(__m256i va0, __m256i va1, __m256i vb0, __m256i vb1) {
    __m256i vmul0 = _mm256_mul_epi32(va0, vb0);
    __m256i vmul1 = _mm256_mul_epi32(va1, vb1);
    __m256i vlow = VIntlv(vmul0, vmul1);
    __m256i vquo0 = _mm256_srli_epi64(_mm256_mul_epi32(_mm256_srli_epi64(vmul0, 29), vbmm), BmW);
    __m256i vquo1 = _mm256_srli_epi64(_mm256_mul_epi32(_mm256_srli_epi64(vmul1, 29), vbmm), BmW);
    __m256i vval0 = _mm256_mul_epi32(vquo0, vm64);
    __m256i vval1 = _mm256_mul_epi32(vquo1, vm64);
    __m256i vval = VIntlv(vval0, vval1);
    __m256i vra = _mm256_sub_epi32(vlow, vval);
    __m256i vrb = _mm256_add_epi32(vra, vm32);
    __m256i vrc = _mm256_sub_epi32(vra, vm32);
    __m256i vmin = _mm256_min_epu32(vra, vrb);
    return _mm256_min_epu32(vmin, vrc);
}

inline __m256i VMul(__m256i va, __m256i vb0, __m256i vb1) {
    return VMul(VEx0(va), VEx1(va), vb0, vb1);
}

inline __m256i VMul(__m256i va, __m256i vb) {
    return VMul(va, VEx0(vb), VEx1(vb));
}

int main() {
    mt19937 rng((uint32_t) system_clock::now().time_since_epoch().count());
    alignas(32) static uint32_t A[8];
    alignas(32) static uint32_t W[8];
    alignas(32) static uint32_t B[8];
    for(;;) {
        for (auto& a : A)
            a = rng() % Mod;
        for (auto& w : W)
            w = rng() % Mod;
        __m256i va = _mm256_load_si256((__m256i*) A);
        __m256i vw = _mm256_load_si256((__m256i*) W);
        __m256i vb = VMul(va, vw);
        _mm256_store_si256((__m256i*) B, vb);
        for (auto i = 0; i < 8; ++i)
            if (B[i] != MMul(A[i], W[i])) {
                cout << "fuck " << i << ": (" << A[i] << ", " << W[i] << ") = " << MMul(A[i], W[i]);
                cout << " != " << B[i] << endl;
                return 0;
            }
    }
}
