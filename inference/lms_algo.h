
#ifndef INFERENCE_LMS_ALGO_H
#define INFERENCE_LMS_ALGO_H

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <stdint.h>
#include <cassert>

using namespace std;

// power of 2 at most x, undefined for x == 0
inline uint32_t bsr(uint32_t x) {
    return 31 - __builtin_clz(x);
}

template<typename T>
inline uint32_t binarySearchBranchless(const std::vector<T> &arr, const T key, uint32_t mid, int threshold) {
//    cout << "searching for key = " << key << endl;
    int n = threshold * 2;
    intptr_t pos = -1;
    intptr_t logstep = bsr(n - 1);
    intptr_t step = intptr_t(1) << logstep;
    int start = mid - threshold;
//    for (int i = start; i <= start + n; ++i) {
//        cout << (long) arr[i] << endl;
//    }
//    cout << endl;
//    cout << "start = " << start << endl;
    pos = (arr[start + pos + n - step] < key ? pos + n - step : pos);
//    cout << "pivot = " << arr[start + pos + n - step] << " pos = " << pos << " step = " << step << endl;
    step >>= 1;
    while (step > 0) {
        pos = (arr[start + pos + step] < key ? pos + step : pos);
//        cout << "pivot = " << arr[start + pos + step] << " pos = " << pos << " step = " << step << endl;
        step >>= 1;
    }
    pos += 1;
    int ans = (uint32_t) (arr[start + pos] >= key ? pos : n) + start;
//    cout << "pivot = " << arr[start + pos] << " pos = " << pos << " step = " << step << endl;
    return ans;
}

template<typename T>
inline uint32_t binarySearchBranchless2(const std::vector<T> &arr, const T key, uint32_t start, uint32_t end) {
//    cout << "searching for key = " << key << endl;
    int n = end - start + 1;
    intptr_t pos = -1;
    intptr_t logstep = bsr(n - 1);
    intptr_t step = intptr_t(1) << logstep;
//    int start = mid - threshold;
    pos = (arr[start + pos + n - step] < key ? pos + n - step : pos);
    step >>= 1;
    while (step > 0) {
        pos = (arr[start + pos + step] < key ? pos + step : pos);
        step >>= 1;
    }
    pos += 1;
    int ans = (uint32_t) (arr[start + pos] >= key ? pos : n) + start;
    return ans;
}

#endif //INFERENCE_LMS_ALGO_H
