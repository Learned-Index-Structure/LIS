#include "lms_algo.hpp"
#include "iaca_mac/iacaMarks.h"

// https://gist.github.com/slode/5ce2a6eb9be1b185b584d2b7f3b94422

using namespace std;
typedef std::chrono::high_resolution_clock Clock;

int mask = 0;

int main(int argc, char **argv) {
    uint32_t n = 200000000;
    BinaryInsert(n);

    uint32_t res = 0;
    long r_sum = 0;
    uint32_t niter = 10000000;

    uint32_t i = 0;
    auto t1 = Clock::now();

    while (i++ < niter) {
        uint32_t key = rand() % (n) + 1;
        res = BinarySearch(vec, key, n);

//        if (res != key)
//            cout << "key, pos: " << key << ", " << res << endl;
        r_sum += res;
    }


    auto t2 = Clock::now();

    cout << "pos: " << r_sum << "Time taken: "
         << chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count() / niter
         << "ns" << endl;
    return 0;
}