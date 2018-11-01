#pragma GCC optimize ("Ofast")
#pragma GCC target ("avx2")

#include <immintrin.h>

#include <algorithm>
#include <cstdio>
#include <utility>

using namespace std;

typedef unsigned char U8;
typedef int I32;
typedef unsigned int U32;
typedef long long I64;
typedef unsigned long long U64;

#define MaxBuf 2097152
#define MaxM 100001
#define Mod 998244353
#define G 3
#define GInv 332748118
#define MaxExp 19
#define MaxLen 524288
#define BmM 288737297
#define BmW 29

char IBuf[MaxBuf];
char OBuf[MaxBuf];
char* IPtr = IBuf;
char* OPtr = OBuf;

inline void FetchAll() {
    fread(IBuf, 1, MaxBuf, stdin);
}

inline void FlushAll() {
    fwrite(OBuf, 1, (size_t) (OPtr - OBuf), stdout);
}

template<class Int>
inline Int ReadU() {
    while (*IPtr < '0')
        ++IPtr;
    Int res = (Int) (*IPtr++ & 0x0f);
    while (*IPtr >= '0')
        res = res * 10 + (Int) (*IPtr++ & 0x0f);
    return res;
}

template<class Int>
inline void PrintU(Int val) {
    if (!val)
        *OPtr++ = '0';
    else {
        int cnt = 0;
        while (val) {
            OPtr[cnt++] = (char) ((val % 10) | '0');
            val /= 10;
        }
        reverse(OPtr, OPtr + cnt);
        OPtr += cnt;
    }
    *OPtr++ = '\n';
}

U32 GcdEx(U32 A, U32 B, I32& x, I32& y) {
    if (!B) {
        x = 1;
        y = 0;
        return A;
    }
    U32 d = GcdEx(B, A % B, y, x);
    y -= x * (I32) (A / B);
    return d;
}

inline U32 MAdd(U32 A, U32 B) {
    U32 res = A + B;
    return res < Mod ? res : res - Mod;
}

inline U32 MSub(U32 A, U32 B) {
    U32 res = A - B;
    return A < B ? res + Mod : res;
}

inline U32 MMul(U32 A, U32 B) {
    return (U32) ((U64) A * B % Mod);
}

inline U32 MPow(U32 A, U32 B) {
    U32 res = 1;
    while (B) {
        if (B & 1)
            res = MMul(res, A);
        A = MMul(A, A);
        B >>= 1;
    }
    return res;
}

inline U32 MInv(U32 N) {
    I32 x, y;
    GcdEx(N, Mod, x, y);
    x %= Mod;
    return (U32) (x < 0 ? x + Mod : x);
}

#define VLod(A) (_mm256_load_si256((const __m256i*) (A)))
#define VSto(A, v) (_mm256_store_si256((__m256i*) (A), v))
#define VEx0(v) (_mm256_shuffle_epi8(v, vm0))
#define VEx1(v) (_mm256_shuffle_epi8(v, vm1))
#define VIntlv(v0, v1) (_mm256_blend_epi32(v0, _mm256_shuffle_epi32(v1, 0xb1), 0xaa))
#define VAdd(va, vb, vt) ( \
    vt = _mm256_add_epi32(va, vb), \
    _mm256_min_epu32(_mm256_sub_epi32(vt, vm32), vt) \
)
#define VSub(va, vb, vt) ( \
    vt = _mm256_sub_epi32(va, vb), \
    _mm256_min_epu32(_mm256_add_epi32(vt, vm32), vt) \
)
#define VMul4(va0, va1, vb0, vb1, vt, vu, vv, vw) ( \
    vt = _mm256_mul_epi32(va0, vb0), \
    vu = _mm256_mul_epi32(va1, vb1), \
    vv = _mm256_mul_epi32(_mm256_srli_epi64(_mm256_mul_epi32(_mm256_srli_epi64(vt, 29), vbmm), BmW), vm64), \
    vw = _mm256_mul_epi32(_mm256_srli_epi64(_mm256_mul_epi32(_mm256_srli_epi64(vu, 29), vbmm), BmW), vm64), \
    vt = VIntlv(vt, vu), \
    vu = VIntlv(vv, vw), \
    vt = _mm256_sub_epi32(vt, vu), \
    _mm256_min_epu32(_mm256_min_epu32(vt, _mm256_add_epi32(vt, vm32)), _mm256_sub_epi32(vt, vm32)) \
)
#define VMul3(va, vb0, vb1, vt, vu, vv, vw) ( \
    vt = VEx0(va), \
    vu = VEx1(va), \
    VMul4(vt, vu, vb0, vb1, vt, vu, vv, vw) \
)
#define VMul2(va, vb, vt, vu, vv, vw) ( \
    vv = VEx0(vb), \
    vw = VEx1(vb), \
    VMul3(va, vv, vw, vt, vu, vv, vw) \
)
#if 0
    const __m256i vm0 = _mm256_set_epi64x(
        0x111111111b1a1918, 0x1111111113121110,
        0x111111110b0a0908, 0x1111111103020100
    );
    const __m256i vm1 = _mm256_set_epi64x(
        0x111111111f1e1d1c, 0x1111111117161514,
        0x111111110f0e0d0c, 0x1111111107060504
    );
#endif

inline void VMul(U32* __restrict__ A, U32 Len, U32 W) {
    if (Len < 8) {
        for (U32 i = 0; i < Len; ++i)
            A[i] = MMul(A[i], W);
        return;
    }
    const __m256i vm32 = _mm256_set1_epi32(Mod);
    const __m256i vm64 = _mm256_set1_epi64x(Mod);
    const __m256i vbmm = _mm256_set1_epi64x(BmM);
    const __m256i vm0 = _mm256_set_epi64x(
        0x111111111b1a1918, 0x1111111113121110,
        0x111111110b0a0908, 0x1111111103020100
    );
    const __m256i vm1 = _mm256_set_epi64x(
        0x111111111f1e1d1c, 0x1111111117161514,
        0x111111110f0e0d0c, 0x1111111107060504
    );
    __m256i vta, vtb, vtc, vtd;
    __m256i vw = _mm256_set1_epi64x(W);
    for (U32 i = 0; i < Len; i += 8) {
        __m256i va = VLod(A + i);
        __m256i vr = VMul3(va, vw, vw, vta, vtb, vtc, vtd);
        VSto(A + i, vr);
    }
}

inline void VMul(U32* __restrict__ A, const U32* __restrict__ B, U32 Len) {
    if (Len < 8) {
        for (U32 i = 0; i < Len; ++i)
            A[i] = MMul(A[i], B[i]);
        return;
    }
    const __m256i vm32 = _mm256_set1_epi32(Mod);
    const __m256i vm64 = _mm256_set1_epi64x(Mod);
    const __m256i vbmm = _mm256_set1_epi64x(BmM);
    const __m256i vm0 = _mm256_set_epi64x(
        0x111111111b1a1918, 0x1111111113121110,
        0x111111110b0a0908, 0x1111111103020100
    );
    const __m256i vm1 = _mm256_set_epi64x(
        0x111111111f1e1d1c, 0x1111111117161514,
        0x111111110f0e0d0c, 0x1111111107060504
    );
    __m256i vta, vtb, vtc, vtd;
    for (U32 i = 0; i < Len; i += 8) {
        __m256i va = VLod(A + i);
        __m256i vb = VLod(B + i);
        __m256i vr = VMul2(va, vb, vta, vtb, vtc, vtd);
        VSto(A + i, vr);
    }
}

inline void VSqr(U32* __restrict__ A, U32 Len) {
    if (Len < 8) {
        for (U32 i = 0; i < Len; ++i)
            A[i] = MMul(A[i], A[i]);
        return;
    }
    const __m256i vm32 = _mm256_set1_epi32(Mod);
    const __m256i vm64 = _mm256_set1_epi64x(Mod);
    const __m256i vbmm = _mm256_set1_epi64x(BmM);
    const __m256i vm0 = _mm256_set_epi64x(
        0x111111111b1a1918, 0x1111111113121110,
        0x111111110b0a0908, 0x1111111103020100
    );
    const __m256i vm1 = _mm256_set_epi64x(
        0x111111111f1e1d1c, 0x1111111117161514,
        0x111111110f0e0d0c, 0x1111111107060504
    );
    __m256i vta, vtb, vtc, vtd;
    for (U32 i = 0; i < Len; i += 8) {
        __m256i va = VLod(A + i);
        __m256i v0 = VEx0(va);
        __m256i v1 = VEx1(va);
        __m256i vr = VMul4(v0, v1, v0, v1, vta, vtb, vtc, vtd);
        VSto(A + i, vr);
    }
}

U32 WbFwd[MaxExp + 1];
U32 WbInv[MaxExp + 1];
U32 LenInv[MaxExp + 1];

inline void NttInitAll(int Max) {
    for (int Exp = 0; Exp <= Max; ++Exp) {
        WbFwd[Exp] = MPow(G, (Mod - 1) >> Exp);
        WbInv[Exp] = MPow(GInv, (Mod - 1) >> Exp);
        LenInv[Exp] = MInv(1u << Exp);
    }
}

inline void NttImpl1(U32* __restrict__ A, U32 Len) {
    for (U32 j = 0; j < Len; j += 2) {
        U32 a0 = MAdd(A[j + 0], A[j + 1]);
        U32 b0 = MSub(A[j + 0], A[j + 1]);
        A[j + 0] = a0;
        A[j + 1] = b0;
    }
}

inline void NttFwd2(U32* __restrict__ A, U32 Len, U32 Wn) {
    for (U32 j = 0; j < Len; j += 4) {
        U32 a0 = MAdd(A[j + 0], A[j + 2]);
        U32 a1 = MAdd(A[j + 1], A[j + 3]);
        U32 b0 = MSub(A[j + 0], A[j + 2]);
        U32 b1 = MSub(A[j + 1], A[j + 3]);
        A[j + 0] = a0;
        A[j + 1] = a1;
        A[j + 2] = b0;
        A[j + 3] = MMul(b1, Wn);
    }
}

inline void NttFwd3(U32* __restrict__ A, U32 Len, U32 Wn) {
    U32 W2 = MMul(Wn, Wn);
    U32 W3 = MMul(W2, Wn);
    const __m128i vm32 = _mm_set1_epi32(Mod);
    for (U32 j = 0; j < Len; j += 8) {
        __m128i va = _mm_load_si128((const __m128i*) (A + j));
        __m128i vb = _mm_load_si128((const __m128i*) (A + j + 4));
        __m128i vc = _mm_add_epi32(va, vb);
        __m128i vd = _mm_sub_epi32(va, vb);
        __m128i ve = _mm_sub_epi32(vc, _mm_andnot_si128(_mm_cmpgt_epi32(vm32, vc), vm32));
        __m128i vf = _mm_add_epi32(vd, _mm_and_si128(_mm_cmpgt_epi32(vb, va), vm32));
        _mm_store_si128((__m128i*) (A + j), ve);
        _mm_store_si128((__m128i*) (A + j + 4), vf);
        A[j + 5] = MMul(Wn, A[j + 5]);
        A[j + 6] = MMul(W2, A[j + 6]);
        A[j + 7] = MMul(W3, A[j + 7]);
    }
}

inline void NttFwd(U32* __restrict__ A, int Exp) {
    U32 Len = 1u << Exp;
    U32 Wn = WbFwd[Exp];
    const __m256i vm32 = _mm256_set1_epi32(Mod);
    const __m256i vm64 = _mm256_set1_epi64x(Mod);
    const __m256i vbmm = _mm256_set1_epi64x(BmM);
    const __m256i vm0 = _mm256_set_epi64x(
        0x111111111b1a1918, 0x1111111113121110,
        0x111111110b0a0908, 0x1111111103020100
    );
    const __m256i vm1 = _mm256_set_epi64x(
        0x111111111f1e1d1c, 0x1111111117161514,
        0x111111110f0e0d0c, 0x1111111107060504
    );
    __m256i vta, vtb, vtc, vtd;
    for (int i = Exp - 1; i >= 3; --i) {
        U32 ChkSiz = 1u << i;
        U32 tw2 = MMul(Wn, Wn);
        U32 tw3 = MMul(tw2, Wn);
        U32 tw4 = MMul(tw3, Wn);
        U32 tw5 = MMul(tw4, Wn);
        U32 tw6 = MMul(tw5, Wn);
        U32 tw7 = MMul(tw6, Wn);
        U32 twn = MMul(tw7, Wn);
        __m256i vw32 = _mm256_set_epi32(tw7, tw6, tw5, tw4, tw3, tw2, Wn, 1);
        __m256i vwn = _mm256_set1_epi64x(twn);
        for (U32 j = 0; j < Len; j += 2u << i) {
            U32* A_ = A + j;
            U32* B_ = A_ + ChkSiz;
            __m256i vw = vw32;
            for (U32 k = 0; k < ChkSiz; k += 8) {
                __m256i va = VLod(A_ + k);
                __m256i vb = VLod(B_ + k);
                __m256i vw0 = VEx0(vw);
                __m256i vw1 = VEx1(vw);
                __m256i vc = VAdd(va, vb, vta);
                __m256i vd = VSub(va, vb, vta);
                __m256i ve = VMul3(vd, vw0, vw1, vta, vtb, vtc, vtd);
                VSto(A_ + k, vc);
                VSto(B_ + k, ve);
                vw = VMul4(vw0, vw1, vwn, vwn, vta, vtb, vtc, vtd);
            }
        }
        Wn = MMul(Wn, Wn);
    }
    if (Exp >= 3) {
        NttFwd3(A, Len, Wn);
        Wn = MMul(Wn, Wn);
    }
    if (Exp >= 2)
        NttFwd2(A, Len, Wn);
    if (Exp)
        NttImpl1(A, Len);
}

inline void NttInv2(U32* __restrict__ A, U32 Len, U32 Wn) {
    for (U32 j = 0; j < Len; j += 4) {
        U32 a0 = A[j + 0];
        U32 a1 = A[j + 1];
        U32 b0 = A[j + 2];
        U32 b1 = MMul(A[j + 3], Wn);
        A[j + 0] = MAdd(a0, b0);
        A[j + 1] = MAdd(a1, b1);
        A[j + 2] = MSub(a0, b0);
        A[j + 3] = MSub(a1, b1);
    }
}

inline void NttInv3(U32* __restrict__ A, U32 Len, U32 Wn) {
    U32 W2 = MMul(Wn, Wn);
    U32 W3 = MMul(W2, Wn);
    const __m128i vm32 = _mm_set1_epi32(Mod);
    for (U32 j = 0; j < Len; j += 8) {
        A[j + 5] = MMul(Wn, A[j + 5]);
        A[j + 6] = MMul(W2, A[j + 6]);
        A[j + 7] = MMul(W3, A[j + 7]);
        __m128i va = _mm_load_si128((const __m128i*) (A + j));
        __m128i vb = _mm_load_si128((const __m128i*) (A + j + 4));
        __m128i vc = _mm_add_epi32(va, vb);
        __m128i vd = _mm_sub_epi32(va, vb);
        __m128i ve = _mm_sub_epi32(vc, _mm_andnot_si128(_mm_cmpgt_epi32(vm32, vc), vm32));
        __m128i vf = _mm_add_epi32(vd, _mm_and_si128(_mm_cmpgt_epi32(vb, va),vm32));
        _mm_store_si128((__m128i*) (A + j), ve);
        _mm_store_si128((__m128i*) (A + j + 4), vf);
    }
}

inline void NttInv(U32* __restrict__ A, int Exp) {
    if (!Exp)
        return;
    U32 Len = 1u << Exp;
    NttImpl1(A, Len);
    if (Exp == 1) {
        VMul(A, Len, LenInv[1]);
        return;
    }
    U32 Ws[MaxExp];
    Ws[0] = WbInv[Exp];
    for (int i = 1; i < Exp; ++i)
        Ws[i] = MMul(Ws[i - 1], Ws[i - 1]);
    NttInv2(A, Len, Ws[Exp - 2]);
    if (Exp == 2) {
        VMul(A, Len, LenInv[2]);
        return;
    }
    NttInv3(A, Len, Ws[Exp - 3]);
    if (Exp == 3) {
        VMul(A, Len, LenInv[3]);
        return;
    }
    const __m256i vm32 = _mm256_set1_epi32(Mod);
    const __m256i vm64 = _mm256_set1_epi64x(Mod);
    const __m256i vbmm = _mm256_set1_epi64x(BmM);
    const __m256i vm0 = _mm256_set_epi64x(
        0x111111111b1a1918, 0x1111111113121110,
        0x111111110b0a0908, 0x1111111103020100
    );
    const __m256i vm1 = _mm256_set_epi64x(
        0x111111111f1e1d1c, 0x1111111117161514,
        0x111111110f0e0d0c, 0x1111111107060504
    );
    __m256i vta, vtb, vtc, vtd;
    for (int i = 3; i < Exp; ++i) {
        U32 ChkSiz = 1u << i;
        U32 Wn = Ws[Exp - 1 - i];
        U32 tw2 = MMul(Wn, Wn);
        U32 tw3 = MMul(tw2, Wn);
        U32 tw4 = MMul(tw3, Wn);
        U32 tw5 = MMul(tw4, Wn);
        U32 tw6 = MMul(tw5, Wn);
        U32 tw7 = MMul(tw6, Wn);
        U32 twn = MMul(tw7, Wn);
        __m256i vw32 = _mm256_set_epi32(tw7, tw6, tw5, tw4, tw3, tw2, Wn, 1);
        __m256i vwn = _mm256_set1_epi64x(twn);
        for (U32 j = 0; j < Len; j += 2u << i) {
            U32* A_ = A + j;
            U32* B_ = A_ + ChkSiz;
            __m256i vw = vw32;
            for (U32 k = 0; k < ChkSiz; k += 8) {
                __m256i vw0 = VEx0(vw);
                __m256i vw1 = VEx1(vw);
                __m256i vbl = VLod(B_ + k);
                __m256i vb = VMul3(vbl, vw0, vw1, vta, vtb, vtc, vtd);
                vw = VMul4(vw0, vw1, vwn, vwn, vta, vtb, vtc, vtd);
                __m256i va = VLod(A_ + k);
                __m256i vc = VAdd(va, vb, vta);
                __m256i vd = VSub(va, vb, vta);
                VSto(A_ + k, vc);
                VSto(B_ + k, vd);
            }
        }
    }
    VMul(A, Len, LenInv[Exp]);
}

inline int Log2Ceil(U32 N) {
    static const U8 Table[32] = {
        0,  9,  1,  10, 13, 21, 2,  29,
        11, 14, 16, 18, 22, 25, 3,  30,
        8,  12, 20, 28, 15, 17, 24, 7,
        19, 27, 23, 6,  26, 5,  4,  31,
    };
    N = (N << 1) - 1;
    N |= N >> 1;
    N |= N >> 2;
    N |= N >> 4;
    N |= N >> 8;
    N |= N >> 16;
    return (int) Table[(N * 0x07c4acddu) >> 27];
}

int Exp;
U32 Len;
U32 M;
U32 T[MaxLen];
U32 K[MaxM];
U32 A[MaxLen];

namespace poly {

U32 TA[MaxLen], TB[MaxLen], TC[MaxLen];
U32 FNtt[MaxLen], FRInvNtt[MaxLen];

inline void Inv(U32 T[], const U32 A[], U32 N) {
    if (N == 1) {
        T[0] = MInv(A[0]);
        return;
    }
    U32 M = (N + 1) >> 1;
    Inv(T, A, M);
    int Exp = Log2Ceil(N + M - 1);
    U32 Len = 1u << Exp;
    copy(A, A + N, TA);
    NttFwd(TA, Exp);
    copy(T, T + M, TB);
    NttFwd(TB, Exp);
    for (U32 i = 0; i < Len; ++i)
        TB[i] = MMul(MSub(2, MMul(TA[i], TB[i])), TB[i]);
    NttInv(TB, Exp);
    copy(TB + M, TB + N, T + M);
    fill(TA, TA + Len, 0);
    fill(TB, TB + Len, 0);
}

inline void SetDiv() {
    FNtt[M] = 1;
    for (U32 i = 1; i <= M; ++i)
        FNtt[M - i] = MSub(0, K[i]);
    reverse_copy(FNtt + 1, FNtt + M + 1, TC);
    NttFwd(FNtt, Exp);
    Inv(FRInvNtt, TC, M);
    NttFwd(FRInvNtt, Exp);
}

inline void Rem(U32 A[]) {
    U32 N = M + M;
    reverse_copy(A + M, A + N, TC);
    NttFwd(TC, Exp);
    VMul(TC, FRInvNtt, Len);
    NttInv(TC, Exp);
    reverse(TC, TC + M);
    fill(TC + M, TC + N, 0);
    NttFwd(TC, Exp);
    VMul(TC, FNtt, Len);
    NttInv(TC, Exp);
    for (U32 i = 0; i < M; ++i)
        A[i] = MSub(A[i], TC[i]);
    fill(A + M, A + N, 0);
    fill(TC + M, TC + N, 0);
}

}

void PolyPow(U64 N) {
    if (N < M) {
        T[N] = 1;
        return;
    }
    if (N == M) {
        for (U32 i = 1; i <= M; ++i)
            T[M - i] = K[i];
        return;
    }
    PolyPow(N >> 1);
    NttFwd(T, Exp);
    VSqr(T, Len);
    NttInv(T, Exp);
    if (N & 1) {
        copy_backward(T, T + M + M, T + M + M + 1);
        T[0] = 0;
    }
    poly::Rem(T);
}

int main() {
    FetchAll();
    M = ReadU<U32>();
    U64 L = ReadU<U64>();
    U64 R = ReadU<U64>();
    for (U32 i = 1; i <= M; ++i)
        K[i] = ReadU<U32>();
    Exp = Log2Ceil(M + M);
    Len = 1u << Exp;
    NttInitAll(Log2Ceil(M << 2));
    T[0] = 1;
    for (U32 i = 1; i <= M; ++i)
        T[i] = MSub(0, K[i]);
    poly::Inv(A, T, M + 1);
    reverse(A, A + M);
    A[M] = 0;
    NttFwd(A, Exp);
    poly::SetDiv();
    fill(T, T + M + 1, 0);
    PolyPow(L + M - 1);
    NttFwd(T, Exp);
    VMul(A, T, Len);
    NttInv(A, Exp);
    reverse_copy(A + M, A + M + M - 1, A);
    for (U32 i = M; L + i <= R; ++i) {
        U32 res = 0;
        for (U32 j = 1; j <= M; ++j)
            res = MAdd(res, MMul(A[i - j], K[j]));
        A[i] = res;
    }
    for (U32 i = 0; i <= R - L; ++i)
        PrintU(A[i]);
    FlushAll();
    return 0;
}
