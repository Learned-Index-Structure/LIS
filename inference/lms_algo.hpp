#ifndef INFERENCE_LMS_ALGO_HPP
#define INFERENCE_LMS_ALGO_HPP

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
    int n = threshold * 2;
    intptr_t pos = -1;
    intptr_t logstep = bsr(n - 1);
    intptr_t step = intptr_t(1) << logstep;
    int start = mid - threshold;

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

template<typename T>
inline uint32_t binarySearchBranchless2(const std::vector<T> &arr, const T key, uint32_t start, uint32_t end) {
    int n = end - start + 1;
    intptr_t pos = -1;
    intptr_t logstep = bsr(n - 1);
    intptr_t step = intptr_t(1) << logstep;
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

#endif //INFERENCE_LMS_ALGO_HPP