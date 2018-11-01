#include <cstdint>
#include <random>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

using namespace std;

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

int main() {
    vector<thread> V;
    mutex io;
    for (auto i = 0; i < 8; ++i)
        V.emplace_back([&io]() {
            mt19937 rng(random_device{}());
            {
                unique_lock<mutex> ul(io);
                cout << "Thread starts: " << rng() << endl;
            }
            for (uint32_t A = 0; A < Mod; ++A) {
                auto B = rng() % Mod;
                auto M = MMul(A, B);
                auto W = MWtf(A, B);
                if (M != W) {
                    unique_lock<mutex> ul(io);
                    cout << "(" << A << ", " << B << ") = " << M << " != " << W << endl;
                    break;
                }
            }
        });
    for (auto& t : V)
        t.join();
    return 0;
}
