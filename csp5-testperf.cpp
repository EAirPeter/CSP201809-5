#pragma GCC target("avx2")

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <random>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace std;
using namespace chrono;

constexpr uint32_t Mod = 998244353;
constexpr uint32_t WtfM = 288737297;
constexpr uint32_t WtfW = 29;

constexpr uint32_t MMul(uint32_t A, uint32_t B) {
    return (uint32_t) ((uint64_t) A * B % Mod);
}

constexpr uint32_t MWtf(uint32_t A, uint32_t B) {
    uint64_t mul = (uint64_t) A * B;
    uint32_t low = (uint32_t) mul;
    uint32_t quo = (uint32_t) (((uint64_t) (mul >> 29) * WtfM) >> WtfW);
    uint32_t val = quo * Mod;
    uint32_t tmp = low - val;
    tmp = (int32_t) tmp < 0 ? tmp + Mod : tmp;
    tmp = (int32_t) tmp < 0 ? tmp + Mod : tmp;
    return tmp < Mod ? tmp : tmp - Mod;
}

constexpr size_t Len = 262144;

void AMul(uint32_t* __restrict__ A, uint32_t Wn, size_t N) {
    uint32_t W = 1;
    for (size_t i = 0; i < N; ++i) {
        A[i] = MMul(A[i], W);
        W = MMul(W, Wn);
    }
}

void VMul(uint32_t* __restrict__ A, const uint32_t* __restrict__ B) {
    for (size_t i = 0; i < 4; ++i)
        A[i] = MWtf(A[i], B[i]);
}

void AXWtf(uint32_t* __restrict__ A, uint32_t Wn, size_t N) {
    uint32_t W0[4], W1[4], WN[4];
    W0[0] = 1;
    W1[0] = Wn;
    W0[1] = MMul(Wn, Wn);
    W1[1] = MMul(W0[1], Wn);
    W0[2] = MMul(W1[1], Wn);
    W1[2] = MMul(W0[2], Wn);
    W0[3] = MMul(W1[2], Wn);
    W1[3] = MMul(W0[3], Wn);
    WN[0] = WN[1] = WN[2] = WN[3] = MMul(W1[3], Wn);
    for (size_t i = 0; i < N; i += 8) {
        uint32_t A0[4], A1[4];
        A0[0] = A[i + 0];
        A1[0] = A[i + 1];
        A0[1] = A[i + 2];
        A1[1] = A[i + 3];
        A0[2] = A[i + 4];
        A1[2] = A[i + 5];
        A0[3] = A[i + 6];
        A1[3] = A[i + 7];
        VMul(A0, W0);
        VMul(A1, W1);
        A[i + 0] = A0[0];
        A[i + 1] = A1[0];
        A[i + 2] = A0[1];
        A[i + 3] = A1[1];
        A[i + 4] = A0[2];
        A[i + 5] = A1[2];
        A[i + 6] = A0[3];
        A[i + 7] = A1[3];
        VMul(W0, WN);
        VMul(W1, WN);
    }
}

void AWtf(uint32_t* __restrict__ A, uint32_t Wn, size_t N) {
    auto W0 = uint32_t{1};
    auto W1 = Wn;
    auto W2 = MMul(Wn, Wn);
    auto W3 = MMul(W2, Wn);
    auto W4 = MMul(W3, Wn);
    auto W5 = MMul(W4, Wn);
    auto W6 = MMul(W5, Wn);
    auto W7 = MMul(W6, Wn);
    Wn = MMul(W7, Wn);
    __m256i vw = _mm256_set_epi32(W7, W6, W5, W4, W3, W2, W1, W0);
    __m256i vwn = _mm256_set1_epi64x(Wn);
    __m256i vm0 = _mm256_set_epi64x(
        0x111111111b1a1918,
        0x1111111113121110,
        0x111111110b0a0908,
        0x1111111103020100
    );
    __m256i vm1 = _mm256_set_epi64x(
        0x111111111f1e1d1c,
        0x1111111117161514,
        0x111111110f0e0d0c,
        0x1111111107060504
    );
    __m256i vmagic = _mm256_set1_epi64x(WtfM);
    __m256i vmod = _mm256_set1_epi64x(Mod);
    __m256i vm32 = _mm256_set1_epi32(Mod);
    __m256i vzero = _mm256_set1_epi32(0);
    for (size_t i = 0; i < N; i += 8) {
        __m256i vw0 = _mm256_shuffle_epi8(vw, vm0);
        __m256i vw1 = _mm256_shuffle_epi8(vw, vm1);
        {
            __m256i va = _mm256_load_si256((__m256i*) (A + i));
            __m256i vmul0 = _mm256_mul_epi32(_mm256_shuffle_epi8(va, vm0), vw0);
            __m256i vmul1 = _mm256_mul_epi32(_mm256_shuffle_epi8(va, vm1), vw1);
            __m256i vlow = _mm256_blend_epi32(vmul0, _mm256_shuffle_epi32(vmul1, 0xb1), 0xaa);
            __m256i vquo0 = _mm256_srli_epi64(_mm256_mul_epi32(_mm256_srli_epi64(vmul0, 29), vmagic), WtfW);
            __m256i vquo1 = _mm256_srli_epi64(_mm256_mul_epi32(_mm256_srli_epi64(vmul1, 29), vmagic), WtfW);
            __m256i vval0 = _mm256_mul_epi32(vquo0, vmod);
            __m256i vval1 = _mm256_mul_epi32(vquo1, vmod);
            __m256i vval = _mm256_blend_epi32(vval0, _mm256_shuffle_epi32(vval1, 0xb1), 0xaa);
            __m256i vta = _mm256_sub_epi32(vlow, vval);
            __m256i vtb = _mm256_add_epi32(vta, _mm256_and_si256(_mm256_cmpgt_epi32(vzero, vta), vm32));
            __m256i vtc = _mm256_add_epi32(vtb, _mm256_and_si256(_mm256_cmpgt_epi32(vzero, vtb), vm32));
            __m256i vtd = _mm256_sub_epi32(vtc, _mm256_andnot_si256(_mm256_cmpgt_epi32(vm32, vtc), vm32));
            _mm256_store_si256((__m256i*) (A + i), vtd);
        }
        {
            __m256i vmul0 = _mm256_mul_epi32(vw0, vwn);
            __m256i vmul1 = _mm256_mul_epi32(vw1, vwn);
            __m256i vlow = _mm256_blend_epi32(vmul0, _mm256_shuffle_epi32(vmul1, 0xb1), 0xaa);
            __m256i vquo0 = _mm256_srli_epi64(_mm256_mul_epi32(_mm256_srli_epi64(vmul0, 29), vmagic), WtfW);
            __m256i vquo1 = _mm256_srli_epi64(_mm256_mul_epi32(_mm256_srli_epi64(vmul1, 29), vmagic), WtfW);
            __m256i vval0 = _mm256_mul_epi32(vquo0, vmod);
            __m256i vval1 = _mm256_mul_epi32(vquo1, vmod);
            __m256i vval = _mm256_blend_epi32(vval0, _mm256_shuffle_epi32(vval1, 0xb1), 0xaa);
            __m256i vta = _mm256_sub_epi32(vlow, vval);
            __m256i vtb = _mm256_add_epi32(vta, _mm256_and_si256(_mm256_cmpgt_epi32(vzero, vta), vm32));
            __m256i vtc = _mm256_add_epi32(vtb, _mm256_and_si256(_mm256_cmpgt_epi32(vzero, vtb), vm32));
            vw = _mm256_sub_epi32(vtc, _mm256_andnot_si256(_mm256_cmpgt_epi32(vm32, vtc), vm32));
        }
    }
}

int main() {
    mt19937 rng((uint32_t) system_clock::now().time_since_epoch().count());
    alignas(32) static uint32_t A[Len];
    alignas(32) static uint32_t B[Len];
    for (auto& a : A)
        a = rng() % Mod;
    copy(A, A + Len, B);
    uint32_t Wn = rng() % Mod;
    auto Test = [=](auto f, uint32_t* __restrict__ A, const char* Name) {
        for (auto i = 0; i < 256; ++i)
            f(A, Wn, Len);
        auto t1 = high_resolution_clock::now();
        for (auto i = 0; i < 1024; ++i)
            f(A, Wn, Len);
        auto t2 = high_resolution_clock::now();
        cout << Name << ": " << duration_cast<microseconds>(t2 - t1).count() << " us" << endl;
    };
    Test(&AWtf, B, "AWtf");
    Test(&AMul, A, "AMul");
    if (memcmp(A, B, sizeof(A)))
        cout << "fuck" << endl;
    return 0;
}
