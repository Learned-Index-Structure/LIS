#ifndef INFERENCE_LMS_ALGO_HPP
#define INFERENCE_LMS_ALGO_HPP

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <stdint.h>
#include <cassert>

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
inline int exponentialSearch(const std::vector<T> &arr, const T key, uint32_t start, uint32_t end)
{
    int n = end - start + 1;
    // If x is present at firt location itself

//    if(arr[start] == key)
//        return start;

    // Find range for binary search by
    // repeated doubling
    int i = 1;
    while (i < n && arr[i + start] <= key)
        i = i*2;

//    int initialDiff = end - start;
//    int finalDiff = min(i, n) - (i/2);
//    cout<<"reduced space by "<<initialDiff - finalDiff<<endl;

    //  Call binary search for the found range.
    return binarySearchBranchless2<uint64_t>(arr, key, start + (i/2), start + min(i, n));
}

#endif //INFERENCE_LMS_ALGO_HPP