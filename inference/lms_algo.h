
#ifndef INFERENCE_LMS_ALGO_H
#define INFERENCE_LMS_ALGO_H

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <stdint.h>
#include <cassert>

std::vector<uint32_t> vec;

inline void BinaryInsert(uint32_t n) {
    uint32_t i = 0;
    for (i = 0; i < n; i++) {
        vec.push_back(i + 12);
    }
}

template<typename T, size_t n>
inline uint32_t BinarySearch(const std::vector<T> &arr, const T key, int pos) {

    uint32_t step = (1 << (31 - __builtin_clz(n - 1)));
    // uint32_t pos = arr[step - 1] < key ? n - step - 1 : -1;

    while ((step >>= 1) > 0) {
        pos = (arr[pos + step] < key ? pos + step : pos);
    }
    return pos + 1;
}

#endif //INFERENCE_LMS_ALGO_H
