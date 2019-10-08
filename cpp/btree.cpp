#include "btree/btree_map.h"

#include<iostream>
#include<chrono>
#include <x86intrin.h>

using namespace std;
using namespace chrono;

__inline__ unsigned long long ___rdtsc(void) {
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ((unsigned long long) lo) | (((unsigned long long) hi) << 32);
}

// http://oliveryang.net/2015/09/pitfalls-of-TSC-usage/
// https://software.intel.com/en-us/forums/intel-isa-extensions/topic/280440
unsigned long tacc_rdtscp(int *chip, int *core) {
    unsigned long int x;
    unsigned a, d, c;
    __asm__ volatile("rdtscp" : "=a" (a), "=d" (d), "=c" (c));

    *chip = (c & 0xFFF000) >> 12;

    *core = c & 0xFFF;
    return ((unsigned long) a) | (((unsigned long) d) << 32);;

}

btree::btree_map<int, int> tree;

void btree_find(int key) {
    cout << "Key: " << tree.find(key).key() << endl;
    cout << "Position in tree: " << tree.find(key).position << endl; //Returns the position of the node in the btree where key is stored.

    unsigned long long tstart = __rdtsc();
    int pos = tree.find(key).position;
    int val1 = ((*(tree.find(key).node)).value(pos)).second;
    unsigned long long tend = __rdtsc();

    cout << "Value: " << val1 << " : " << endl;
    cout << "RDTSC: " << (tend - tstart) * 0.36873156342 << endl;

    // cout << "Key: " << tree.find(1223333).key() << "  Position in tree: " <<  tree.find(1223333).position << endl;

}

void btree_insert() {
    int start_index = 2000000;
    int end_index = 4000000;

    for (int i = start_index; i < end_index; i++) {
        tree.insert(std::pair<int, int>(i, i + 10));
    }

}
int main() {
    btree_insert();
    int key = 3802030;
    btree_find(key);

    return 0;
}