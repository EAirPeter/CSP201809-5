#include <cinttypes>
#include <cstdint>
#include <fstream>
#include <random>

using namespace std;

int main() {
    ofstream fo("t.txt");
    mt19937 rng{random_device{}()};
    int n = 1000000;
    fo << n << ' ' << n << endl;
    for (int i = 0; i <= n; ++i)
        fo << rng() % 10 << ' ';
    fo << endl;
    for (int i = 0; i <= n; ++i)
        fo << rng() % 10 << ' ';
    fo << endl;
    return 0;
}
