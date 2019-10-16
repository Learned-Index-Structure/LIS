
#include<iostream>
#include<chrono>
#include <x86intrin.h>
#include "btree.h"

using namespace std;
using namespace chrono;

#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

int main() {
    btree_insert(200000000, 400000000);
    int key = 308020300;

    int start_index = 200000000;
    int end_index = 203000000;
    long long sum = 0;
    int val = 0;
    //    auto t1 = Clock::now();
    unsigned long long tstart = __rdtsc();
    for (int i = start_index; i < end_index; i++) {
        val = btree_find(i);
        sum += val;
    }
    unsigned long long tend = __rdtsc();
//    auto t2 = Clock::now();
    cout << "Sum: " << sum << endl;
    cout << "Time: " << (tend - tstart) / 2.7 / (end_index - start_index) << "ns" <<  endl;
//    cout << "Time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()/ (end_index - start_index) << "ns" <<  endl;

    return 0;
}