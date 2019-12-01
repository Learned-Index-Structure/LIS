#ifndef INFERENCE_LMS_ALGO_HPP
#define INFERENCE_LMS_ALGO_HPP

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <stdint.h>
#include <cassert>
#include <algorithm>

#pragma once

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
    start = (start < 0) ? 0 : start;
    uint64_t v = pos + n - step;
    pos = (arr[start + v] < key ? v : pos);
//    cout << "pivot = " << arr[v + start]  << " pos = " << pos << " step = " << step << endl;
    step >>= 1;
    uint32_t t;
    while (step > 0) {
        t = step + pos;
        pos = (arr[t + start] < key) ? t : pos;
//        cout << "pivot = " << arr[t + start]  << " pos = " << pos << " step = " << step << endl;
        step >>= 1;
    }
//    pos += 1;
    int ans = (uint32_t) (arr[start + pos + 1] >= key ? pos + 1 : n) + start;
//    cout << "Last pivot = " << arr[start + pos + 1]   << " pos = " << pos << " step = " << step << endl;
    return ans;
}

template<typename T>
inline uint32_t binarySearchBranchless2(const std::vector<T> &arr, const T key, uint32_t start, uint32_t end) {
    int n = end - start + 1;
    intptr_t pos = -1;
    intptr_t logstep = bsr(n - 1);
    intptr_t step = intptr_t(1) << logstep;
    uint64_t v = pos + n - step;
    pos = (arr[start + v] < key) ? v : pos;
//    cout << "pivot = " << arr[v + start]  << " pos = " << pos << " step = " << step << endl;
    step >>= 1;
    uint32_t t;
    while (step > 0) {
        t = step + pos;
        pos = (arr[t + start] < key ? t : pos);
//        cout << "pivot = " << arr[t + start]  << " pos = " << pos << " step = " << step << endl;
        step >>= 1;
    }
    int ans = (uint32_t) (arr[start + pos + 1] >= key ? pos + 1 : n) + start;
//    cout << "Last pivot = " << arr[start + pos + 1]   << " pos = " << pos << " step = " << step << endl;
    return ans;
}



template<typename T>
inline int exponentialSearch(const std::vector<T> &arr, const T key, uint32_t midPoint, uint32_t leftMargin, uint32_t rightMargin)
{

    uint32_t leftN = midPoint - leftMargin;
    uint32_t rightN = rightMargin - midPoint;

    uint32_t start, end;

    //start exponential search from mid in both the directions
    uint32_t i = 1;
    if (arr[midPoint] == key) {
        return midPoint;
    } else if (arr[midPoint] < key) {
        while (arr[midPoint + i] <= key) {
            i *= 2;
        }
        start = midPoint + (i / 2);
        end = midPoint + min(i, rightN);
    } else {
        while (arr[midPoint - i] >= key) {
            i *= 2;
        }
        start = midPoint - max(i, leftN);
        end = midPoint - (i/2);
    }

    //  Call binary search for the found range.
    return binarySearchBranchless2<uint64_t>(arr, key, start, end);
}

#endif //INFERENCE_LMS_ALGO_HPP