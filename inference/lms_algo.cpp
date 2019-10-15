#include "lms_algo.h"

// https://gist.github.com/slode/5ce2a6eb9be1b185b584d2b7f3b94422

inline uint32_t bitScanReverse(uint32_t x) {
    return 31 - __builtin_clz(x);
}

inline uint32_t bsearch(const vector<uint32_t> &arr, const uint32_t key, const uint32_t n) {

    uint32_t step = (1 << bitScanReverse(n - 1));
    uint32_t pos = arr[step - 1] < key ? n - step - 1 : -1;

    while ((step >>= 1) > 0) {
        pos = (arr[pos + step] < key ? pos + step : pos);
    }
    return pos + 1;
}

int mask = 0;

int main(int argc, char **argv) {
    vector<uint32_t> vec;
    uint32_t n = 200000000;
    uint32_t i = 0;

    for (i = 0; i < n; i++) {
        vec.push_back(i);
    }

    uint32_t res = 0;
    long r_sum = 0;
    uint32_t niter = 10000000;

    i = 0;
    auto t1 = Clock::now();
    while (i++ < niter) {
//        uint32_t key = rand() % (n) + 1;
        uint32_t key = (i + mask) % n;
        res = bsearch(vec, key, n);
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