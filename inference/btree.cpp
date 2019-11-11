
#include<iostream>
#include<chrono>
#include <x86intrin.h>
#include "btree.h"

using namespace std;
using namespace chrono;

#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

int main() {
    btree::btree_map<float, uint32_t> btree;
    vector<float> keys;
    vector<uint64_t> values;

    for(int i = 0 ;i < 100; i++) {
        keys.push_back(100 + i);
        values.push_back(100001 + i);
    }

    btree_insert(btree, keys, values, 0, 100);

    for(int i = 0 ; i < 100 ; i+=5){
        cout << (100 + i) << ", " << btree_find(btree, 100 + i) << endl;
    }

}