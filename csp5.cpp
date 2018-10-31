#include "Io.hpp"
#include "PolyArith.hpp"

using namespace std;

constexpr size_t MaxM = 100001;
constexpr int MaxExp = Log2Ceil(MaxM << 2);
constexpr size_t MaxLen = 1u << MaxExp;
constexpr uint32_t Mod = 998244353;
constexpr uint32_t G = 3;
using MA = ModArith<uint32_t, uint64_t>;

Ntt<MaxExp, uint32_t, uint64_t, Mod, G> NTT;
PolyArith<decltype(NTT)> PA{NTT};

uint32_t T[MaxLen];
size_t M;
int Exp;

void PolyPow(uint64_t N) noexcept {
    if (!N) {
        T[0] = 1;
        return;
    }
    PolyPow(N >> 1);
    NTT.Forward(T, Exp);
    auto Len = size_t{1} << Exp;
    for (size_t i = 0; i < Len; ++i)
        T[i] = MA::Mul(T[i], T[i], Mod);
    NTT.Inverse(T, Exp);
    if (N & 1) {
        for (size_t i = M + M; i; --i)
            T[i] = T[i - 1];
        T[0] = 0;
    }
    PA.Rem(T, T);
    fill(T + M, T + M + M, 0);
}

uint32_t K[MaxM];
uint32_t A[MaxLen];

int main() {
    M = ReadU<size_t>();
    auto L = ReadU<uint64_t>();
    auto R = ReadU<uint64_t>();
    NTT.InitAll(Log2Ceil(M << 2));
    Exp = Log2Ceil(M + M - 1);
    for (size_t i = 1; i <= M; ++i)
        K[i] = ReadU<uint32_t>();
    T[0] = 1;
    for (size_t i = 1; i <= M; ++i)
        T[i] = MA::Sub(0, K[i], Mod);
    PA.Inv(A, T, M + 1);
    reverse(A, A + M);
    A[M] = 0;
    NTT.Forward(A, Exp);
    T[M] = 1;
    for (size_t i = 1; i <= M; ++i)
        T[M - i] = MA::Sub(0, K[i], Mod);
    PA.SetDivisor(M + M, T, M + 1);
    fill(T, T + M + 1, 0);
    PolyPow(L + M - 1);
    NTT.Forward(T, Exp);
    auto Len = size_t{1} << Exp;
    for (size_t i = 0; i < Len; ++i)
        A[i] = MA::Mul(A[i], T[i], Mod);
    NTT.Inverse(A, Exp);
    reverse_copy(A + M, A + M + M - 1, A);
    for (size_t i = M; L + i <= R; ++i) {
        uint32_t res = 0;
        for (size_t j = 1; j <= M; ++j)
            res = MA::Add(res, MA::Mul(A[i - j], K[j], Mod), Mod);
        A[i] = res;
    }
    for (size_t i = 0; i <= R - L; ++i) {
        PrintU(A[i]);
        IoPutChar('\n');
    }
    return 0;
}
