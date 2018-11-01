#include <cstdint>
#include <iostream>

using namespace std;

constexpr auto Low = 29;
constexpr auto C = (119ull << 23) + 1;
constexpr auto A = (C - 1) * (C - 1) >> Low;

int main() {
    for (int w = 0; w + Low < 64; ++w) {
        auto kend = (C * (1ull << w) + A - 1) / A;
        auto ex = 1ull << (w + Low);
        auto k = C - ex % C;
        if (k < kend) {
            auto m = (ex + k) / C;
            if (m * C != ex + k)
                cerr << "fuck" << endl;
            cout << "A = " << A << endl;
            cout << "C = " << C << endl;
            cout << "Low = " << Low << endl;
            cout << "w = " << w << endl;
            cout << "k = " << k << endl;
            cout << "m = " << m << endl;
            cout << "kend = " << kend << endl;
            getchar();
        }
    }
    return 0;
}
