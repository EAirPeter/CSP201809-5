#include <algorithm>
#include <utility>

using namespace std;

#include <stdio.h>

#ifdef _MSC_VER
#define IoGetChar _getchar_nolock
#define IoPutChar _putchar_nolock
#else
#define IoGetChar getchar
#define IoPutChar putchar
#endif

typedef unsigned char U8;
typedef int I32;
typedef unsigned int U32;
typedef long long I64;
typedef unsigned long long U64;

#define MaxBuf 2097152
#define MaxM 100001
#define Mod 998244353
#define G 3
#define MaxExp 19
#define MaxLen 524288

char IBuf[MaxBuf];
char OBuf[MaxBuf];
char* IPtr = IBuf;
char* OPtr = OBuf;

template<class Int>
inline Int ReadU() {
    int ch = IoGetChar();
    while (ch < '0')
        ch = IoGetChar();
    Int res = (Int) (ch & 0x0f);
    ch = IoGetChar();
    while (ch >= '0') {
        res = res * 10 + (Int) (ch & 0x0f);
        ch = IoGetChar();
    }
    return res;
}

template<class Int>
inline void PrintU(Int val) {
    if (!val) {
        IoPutChar('0');
        return;
    }
    int Buf[20];
    int cnt = 0;
    while (val) {
        Buf[cnt++] = (int) (val % 10) | '0';
        val /= 10;
    }
    while (cnt--)
        IoPutChar(Buf[cnt]);
}

inline U32 Gcd(U32 A, U32 B) {
    while (B) {
        U32 C = A % B;
        A = B;
        B = C;
    }
    return A;
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

U32 WFwd[MaxLen << 1];
U32 WInv[MaxLen << 1];
U32 LenInv[MaxExp + 1];

inline void NttGenW(U32 WFwd[], U32 WInv[], int Exp) {
    U32 Len = 1u << Exp;
    U32 fwd = MPow(G, (Mod - 1) >> Exp);
    WFwd[0] = 1;
    for (U32 i = 1; i < Len; ++i)
        WFwd[i] = MMul(WFwd[i - 1], fwd);
    U32 inv = MInv(fwd);
    WInv[0] = 1;
    for (U32 i = 1; i < Len; ++i)
        WInv[i] = MMul(WInv[i - 1], inv);
}

inline void NttInitAll(int Max) {
    for (int Exp = 0; Exp <= Max; ++Exp) {
        LenInv[Exp] = MInv(1u << Exp);
        NttGenW(WFwd + (1 << Exp), WInv + (1 << Exp), Exp);
    }
}

inline void NttImpl(U32 A[], int Exp, const U32 W[]) {
    U32 Len = 1u << Exp;
    for (U32 i = 0, j = 0; i < Len; ++i) {
        if (i > j)
            swap(A[i], A[j]);
        for (U32 k = Len >> 1; (j ^= k) < k; k >>= 1);
    }
    for (int i = 0; i < Exp; ++i) {
        U32 ChkSiz = 1u << i;
        for (U32 j = 0; j < Len; j += 2u << i)
            for (U32 k = 0; k < ChkSiz; ++k) {
                U32 t = MMul(W[(Len >> (i + 1)) * k], A[j + k + ChkSiz]);
                A[j + k + ChkSiz] = MSub(A[j + k], t);
                A[j + k] = MAdd(A[j + k], t);
            }
    }
}

inline void NttFwd(U32 A[], int Exp) {
    NttImpl(A, Exp, WFwd + (1u << Exp));
}

inline void NttInv(U32 A[], int Exp) {
    NttImpl(A, Exp, WInv + (1 << Exp));
    U32 Len = 1u << Exp;
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

namespace poly {

U32 TA[MaxLen], TB[MaxLen], TC[MaxLen];
U32 FNtt[MaxLen], FRInvNtt[MaxLen];
U32 N, M, R;
int ExpN, ExpR;

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
    fill(TA + N, TA + Len, 0);
    NttFwd(TA, Exp);
    copy(T, T + M, TB);
    fill(TB + M, TB + Len, 0);
    NttFwd(TB, Exp);
    for (U32 i = 0; i < Len; ++i)
        TB[i] = MMul(MSub(2, MMul(TA[i], TB[i])), TB[i]);
    NttInv(TB, Exp);
    copy(TB + M, TB + N, T + M);
}

inline void SetDiv(U32 N_, const U32 F[], U32 M_) {
    N = N_;
    M = M_;
    R = N - M + 1;
    ExpN = Log2Ceil(N);
    ExpR = Log2Ceil(R + R - 1);
    copy(F, F + M, FNtt);
    fill(FNtt + M, FNtt + (1 << ExpN), 0);
    NttFwd(FNtt, ExpN);
    U32 cnt = min(M, R);
    reverse_copy(F + M - cnt, F + M, TC);
    fill(TC + cnt, TC + R, 0);
    Inv(FRInvNtt, TC, R);
    fill(FRInvNtt + R, FRInvNtt + (1 << ExpR), 0);
    NttFwd(FRInvNtt, ExpR);
}

inline void Rem(U32 H[], const U32 A[]) {
    U32 LenN = 1u << ExpN;
    U32 LenR = 1u << ExpR;
    U32 cnt = min(N, R);
    reverse_copy(A + N - cnt, A + N, TC);
    fill(TC + cnt, TC + LenR, 0);
    NttFwd(TC, ExpR);
    for (U32 i = 0; i < LenR; ++i)
        TC[i] = MMul(TC[i], FRInvNtt[i]);
    NttInv(TC, ExpR);
    reverse(TC, TC + R);
    fill(TC + R, TC + LenN, 0);
    NttFwd(TC, ExpN);
    for (U32 i = 0; i < LenN; ++i)
        TC[i] = MMul(TC[i], FNtt[i]);
    NttInv(TC, ExpN);
    for (U32 i = 0; i < M - 1; ++i)
        H[i] = MSub(A[i], TC[i]);
}

}

U32 T[MaxLen];
U32 M;
int Exp;
U32 Len;

void PolyPow(U64 N) {
    if (!N) {
        T[0] = 1;
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
    poly::Rem(T, T);
    fill(T + M, T + M + M, 0);
}

U32 K[MaxM];
U32 A[MaxLen];

int main() {
    
    M = ReadU<U32>();
    U64 L = ReadU<U64>();
    U64 R = ReadU<U64>();
    NttInitAll(Log2Ceil(M << 2));
    Exp = Log2Ceil(M + M - 1);
    Len = 1u << Exp;
    for (U32 i = 1; i <= M; ++i)
        K[i] = ReadU<U32>();
    T[0] = 1;
    for (U32 i = 1; i <= M; ++i)
        T[i] = MSub(0, K[i]);
    poly::Inv(A, T, M + 1);
    reverse(A, A + M);
    A[M] = 0;
    NttFwd(A, Exp);
    T[M] = 1;
    for (U32 i = 1; i <= M; ++i)
        T[M - i] = MSub(0, K[i]);
    poly::SetDiv(M + M, T, M + 1);
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
    for (U32 i = 0; i <= R - L; ++i) {
        PrintU(A[i]);
        IoPutChar('\n');
    }
    return 0;
}
