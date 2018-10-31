#pragma GCC optimize ("Ofast")
#pragma GCC target ("avx2")

#include <windows.h>

#include <cstdio>

struct TimeMe {
    ::LARGE_INTEGER qpf;
    ::LARGE_INTEGER qpc;
    TimeMe() {
        ::QueryPerformanceFrequency(&qpf);
        ::QueryPerformanceCounter(&qpc);
    }
    ~TimeMe() {
        ::LARGE_INTEGER now;
        ::QueryPerformanceCounter(&now);
        ::std::fprintf(stderr, "%.6f s\n", (double) (now.QuadPart - qpc.QuadPart) / (double) qpf.QuadPart);
    }
};

TimeMe ImplTimeMe;

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

inline void NttFwd(U32 A[], int Exp) {
    U32 Len = 1u << Exp;
    U32 Wn = WbFwd[Exp];
    for (int i = Exp - 1; i >= 0; --i) {
        U32 ChkSiz = 1u << i;
        for (U32 j = 0; j < Len; j += 2u << i) {
            U32 W = 1;
            for (U32 k = j; k < j + ChkSiz; ++k) {
                U32 a = A[k];
                U32 b = A[k + ChkSiz];
                A[k] = MAdd(a, b);
                A[k + ChkSiz] = MMul(W, MSub(a, b));
                W = MMul(W, Wn);
            }
        }
        Wn = MMul(Wn, Wn);
    }
}

inline void NttInv(U32 A[], int Exp) {
    U32 Len = 1u << Exp;
    U32 Ws[MaxExp];
    Ws[0] = WbInv[Exp];
    for (int i = 1; i < Exp; ++i)
        Ws[i] = MMul(Ws[i - 1], Ws[i - 1]);
    for (int i = 0; i < Exp; ++i) {
        U32 ChkSiz = 1u << i;
        U32 Wn = Ws[Exp - 1 - i];
        for (U32 j = 0; j < Len; j += 2u << i) {
            U32 W = 1;
            for (U32 k = j; k < j + ChkSiz; ++k) {
                U32 a = A[k];
                U32 b = MMul(W, A[k + ChkSiz]);
                A[k] = MAdd(a, b);
                A[k + ChkSiz] = MSub(a, b);
                W = MMul(W, Wn);
            }
        }
    }
    for (U32 i = 0; i < Len; ++i)
        A[i] = MMul(A[i], LenInv[Exp]);
}

inline int Log2(U32 N) {
    static const U8 Table[32] = {
        0,  9,  1,  10, 13, 21, 2,  29,
        11, 14, 16, 18, 22, 25, 3,  30,
        8,  12, 20, 28, 15, 17, 24, 7,
        19, 27, 23, 6,  26, 5,  4,  31,
    };
    N |= N >> 1;
    N |= N >> 2;
    N |= N >> 4;
    N |= N >> 8;
    N |= N >> 16;
    return (int) Table[(N * 0x07c4acddu) >> 27];
}

inline int Log2Ceil(U32 N) {
    return Log2((N << 1) - 1);
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
    for (U32 i = 0; i < Len; ++i)
        TC[i] = MMul(TC[i], FRInvNtt[i]);
    NttInv(TC, Exp);
    reverse(TC, TC + M);
    fill(TC + M, TC + N, 0);
    NttFwd(TC, Exp);
    for (U32 i = 0; i < Len; ++i)
        TC[i] = MMul(TC[i], FNtt[i]);
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
    for (U32 i = 0; i < Len; ++i)
        T[i] = MMul(T[i], T[i]);
    NttInv(T, Exp);
    if (N & 1) {
        for (U32 i = M + M; i; --i)
            T[i] = T[i - 1];
        T[0] = 0;
    }
    poly::Rem(T);
}

int main() {
#ifndef LOCAL_TEST
    freopen("D:\\code\\algo\\std14\\j.txt", "r", stdin);
    freopen("D:\\code\\algo\\std14\\p.txt", "w", stdout);
#endif
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
    for (U32 i = 0; i < Len; ++i)
        A[i] = MMul(A[i], T[i]);
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
