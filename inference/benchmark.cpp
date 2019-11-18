// g++ -O3 -mavx2 -lpthread -std=c++11 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -o bench benchmark.cpp && ./bench
// g++-8 -o bench -O3 -mavx -pthread benchmark.cpp -Lbenchmark/build/src -lbenchmark && ./bench
// g++ -O3 -mavx2 -lpthread -std=c++11 -isystem -Lbenchmark/include  -lbenchmark -o bench benchmark.cpp && ./bench

#include <iostream>
#include "btree.hpp"
#include "lms_algo.hpp"
#include "inference.hpp"
#include "inference.cpp"
#include "/usr/local/include/benchmark/benchmark.h"

long long sum = 0;
string path = "/Users/deepak/Downloads/weights/";

//Ref: https://www.youtube.com/watch?v=nXaxk27zwlk
static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void clobber() {
    asm volatile("" : : : "memory");
}

inline uint32_t GetRandKey(uint32_t sz) {
//    random_device rd;
//    mt19937 gen(rd());
//    uniform_int_distribution<> dis(0, sz);
    return (uint32_t) (rand() % sz);
}

static void RandKeyGen(benchmark::State &state) {
    uint32_t pred = 0;
    for (auto _ : state) {
        escape(&pred);
        pred = GetRandKey(2000000);
        clobber();
    }
}

bool Inference_100000_128_WebLog_Load = true;

static void WebLog_Inference_100000_128(benchmark::State &state) {
    if (Inference_100000_128_WebLog_Load) {
        Inference_100000_128_WebLog_Load = false;
        cleanup();
        setup(path, "weblog", "100000", "128");
        getKeyList(tData, dataLines, maxKey);
    }
    double keyToSearch;
    uint32_t sz = keyList.size() - 1;
    uint32_t i = 0;
    for (auto _ : state) {
        escape(&sum);
        i = GetRandKey(sz);
        keyToSearch = keyList[i];
        key.m[0][0] = keyToSearch;
        sum += infer(keyListInt[i]);
//        cout << sum  << endl;
        clobber();
    }
}

static bool Inference_100000_128_Maps_Load = true;

static void Maps_Inference_100000_128(benchmark::State &state) {
    if (Inference_100000_128_Maps_Load) {
        Inference_100000_128_Maps_Load = false;
        cleanup();
        setup(path, "maps", "100000", "128");
        getKeyList(tData, dataLines, maxKey);
    }
    double keyToSearch;
    uint32_t sz = keyList.size() - 1;
    uint32_t i = 0;
    for (auto _ : state) {
        escape(&sum);
        i = GetRandKey(sz);
        keyToSearch = keyList[i];
        key.m[0][0] = keyToSearch;
        sum += infer(keyListInt[i]);
        clobber();
    }
}

static bool Inference_100000_128_Lognormal_Load = true;

static void Lognormal_Inference_100000_128(benchmark::State &state) {
    if (Inference_100000_128_Lognormal_Load) {
        Inference_100000_128_Lognormal_Load = false;
        cleanup();
        setup(path, "lognormal", "100000", "128");
        getKeyList(tData, dataLines, maxKey);
    }
    double keyToSearch;
    uint32_t sz = keyList.size() - 1;
    uint32_t i = 0;
    for (auto _ : state) {
        escape(&sum);
        i = GetRandKey(sz);
        keyToSearch = keyList[i];
        key.m[0][0] = keyToSearch;
        sum += infer(keyListInt[i]);
        clobber();
    }
}

static bool MAPS_BTREE_128_LOAD = true;
tree_type btree_maps;

static void MAPS_BTREE_128(benchmark::State &state) {
    if (MAPS_BTREE_128_LOAD) {
        MAPS_BTREE_128_LOAD = false;
        cleanup();
        setup(path, "maps", "100000", "128");
        cout << tData.size() << " " << indices.size() << endl;
        btree_insert(btree_maps, tData, indices, 0, tData.size() - 1);
        cout << tData.size() << " " << indices.size() << endl;
    }

    uint32_t sz = tData.size() - 1;
    uint32_t i = 0;
    for (auto _ : state) {
        escape(&sum);
        i = GetRandKey(sz);
        sum += btree_find(btree_maps, tData[i]);
        clobber();
    }
}

static bool LOGNORMAL_BTREE_128_LOAD = true;
tree_type btree_lognormal;

static void LOGNORMAL_BTREE_128(benchmark::State &state) {
    if (LOGNORMAL_BTREE_128_LOAD) {
        LOGNORMAL_BTREE_128_LOAD = false;
        cleanup();
        setup(path, "lognormal", "100000", "128");
        btree_insert(btree_lognormal, tData, indices, 0, tData.size() - 1);
        cout << "btree lognormal size: " << btree_lognormal.size() << endl;
    }

    uint32_t sz = tData.size() - 1;
    uint32_t i = 0;
    for (auto _ : state) {
        escape(&sum);
        i = GetRandKey(sz);
        sum += btree_find(btree_lognormal, tData[i]);
        clobber();
    }
}


BENCHMARK(RandKeyGen);

//BENCHMARK(WebLog_Inference_100000_128);

//BENCHMARK(Maps_Inference_100000_128);

BENCHMARK(Lognormal_Inference_100000_128);

//BENCHMARK(MAPS_BTREE_128);
BENCHMARK(LOGNORMAL_BTREE_128);

BENCHMARK_MAIN();