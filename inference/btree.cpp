#include<iostream>
#include<chrono>
#include <x86intrin.h>
#include <random>
#include "btree.hpp"

using namespace std;
using namespace chrono;


#include <chrono>
#include <fstream>

typedef std::chrono::high_resolution_clock Clock;

#define NUM_ITERS 10000
#define NO_OF_KEYS 10000

vector<uint32_t> getKeyList(uint32_t lines) {
    vector<uint32_t> keys;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, lines);
    for (int i = 0; i < NO_OF_KEYS; ++i) {
        uint64_t k = dis(gen);
        keys.push_back(k);
    }
    return keys;
}

inline
void checkAccuracy(tree_type &btree, vector<uint64_t> keys, vector<uint32_t> values, vector<uint32_t> keyList) {
    for (int i = 0; i < keyList.size(); i++) {
        uint32_t pos = btree_find(btree, keys[keyList[i]]);
        if (!((values[keyList[i]] == pos) || (keys[pos] == keys[keyList[i]]))) {
            cout << "Actual, Found:: " << values[keyList[i]] << ", " << btree_find(btree, keys[keyList[i]]) << endl;
            assert(false);
        }
    }
}

int main() {
    btree::btree_map<uint64_t, uint32_t> btree;
    vector<uint64_t> keys;
    vector<uint32_t> values;
    string file = "/Users/deepak/Downloads/WebLogs/sorted_keys_non_repeated.csv";

    ifstream dataFile(file);
    uint32_t lines = 0;
    double key;
    uint32_t val;
    double offset;// = 1425168000107.1;
    double temp1;

    if (dataFile.is_open()) {
        dataFile >> lines >> temp1 >> temp1;

        for (int i = 0; i < lines; i++) {
            dataFile >> val >> key;
            if (i == 0) offset = key;
            keys.push_back((uint64_t) (key - offset) * 100);
            values.push_back(val);
        }
    }

    btree_insert(btree, keys, values, 0, lines - 1);

    cout << "btree size(): " << btree.size() << ", " << endl;

    vector<uint32_t> keyList = getKeyList(lines);

    //Accuracy Test
    checkAccuracy(btree, keys, values, keyList);


    //Benchmark
    uint64_t sum = 0;

    auto t1 = Clock::now();
    for (int j = 0; j < NUM_ITERS; j++) {
        for (int i = 0; i < keyList.size(); i++) {
            sum += btree_find(btree, keys[keyList[i]]);
        }
    }
    auto t2 = Clock::now();

    cout << "sum = " << sum << endl;
    std::cout << "Time: "
              << chrono::duration<int64_t, std::nano>(t2 - t1).count() / NUM_ITERS / NO_OF_KEYS
              << " nanoseconds" << std::endl;
    return 0;
}


