#include<iostream>
#include<chrono>
#include <x86intrin.h>
#include "btree.h"

using namespace std;
using namespace chrono;


#include <chrono>
#include <fstream>

typedef std::chrono::high_resolution_clock Clock;

#define NUM_ITERS 1
#define NO_OF_KEYS 500000

vector<uint32_t> getKeyList(uint32_t lines) {
    vector<uint32_t> keys;

    for (int i = 0; i < NO_OF_KEYS; ++i) {
        keys.push_back((std::rand() % lines));
    }
    return keys;
}

inline
void checkAccuracy(tree_type &btree, vector<double> keys, vector<uint32_t> values, vector<uint32_t> keyList) {
    for (int i = 0; i < keyList.size(); i++) {
        if (values[keyList[i]] != btree_find(btree, keys[keyList[i]])) {
            cout << "Actual, Found:: " << values[keyList[i]] << ", " << btree_find(btree, keys[keyList[i]]) << endl;
            assert(false);
        }
    }

}

int main() {
    btree::btree_map<double, uint32_t> btree;
    vector<double> keys;
    vector<uint32_t> values;
    string file = "/Users/deepak/Downloads/WebLogs/sorted_keys_non_repeated.csv";

    ifstream dataFile(file);
    uint32_t lines = 0;
    double key;
    uint32_t val;
    double offset = 1425168000107.1;

    if (dataFile.is_open()) {
        dataFile >> lines;

        for (int i = 0; i < lines; i++) {
            dataFile >> val >> key;
            keys.push_back(key - offset);
            values.push_back(val);
        }
    }

    btree_insert(btree, keys, values, 0, lines - 1);

    cout << "btree size(): " << btree.size() << ", " << endl;

    vector<uint32_t> keyList = getKeyList(lines);

//    checkAccuracy(btree, keys, values, keyList);

    uint64_t sum = 0;

    auto t1 = Clock::now();
    for (int i = 0; i < keyList.size(); i++) {
        sum += btree_find(btree, keys[keyList[i]]);
    }
    auto t2 = Clock::now();

    cout << "sum = " << sum << endl;
    std::cout << "Time: "
              << chrono::duration<int64_t, std::nano>(t2 - t1).count() / NUM_ITERS / NO_OF_KEYS
              << " nanoseconds" << std::endl;
    return 0;
}


