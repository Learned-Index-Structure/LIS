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

static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void clobber() {
    asm volatile("" : : : "memory");
}

inline uint32_t GetRandKey(uint32_t sz) {
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


// 10k Models


bool WebLog_Inference_10000_32_Load = true;

static void WebLog_Inference_10000_32(benchmark::State &state) {
    if (WebLog_Inference_10000_32_Load) {
        WebLog_Inference_10000_32_Load = false;
        cleanup(true);
        setup(path, "weblogs", "10000", "32", false);
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

bool WebLog_Inference_10000_64_Load = true;

static void WebLog_Inference_10000_64(benchmark::State &state) {
    if (WebLog_Inference_10000_64_Load) {
        WebLog_Inference_10000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "weblogs", "10000", "64", true);
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

bool Inference_10000_128_WebLog_Load = true;

static void WebLog_Inference_10000_128(benchmark::State &state) {
    if (Inference_10000_128_WebLog_Load) {
        Inference_10000_128_WebLog_Load = false;
        cleanup(false);
        setup(path, "weblogs", "10000", "128", true);
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

bool WebLog_Inference_10000_256_Load = true;

static void WebLog_Inference_10000_256(benchmark::State &state) {
    if (WebLog_Inference_10000_256_Load) {
        WebLog_Inference_10000_256_Load = false;
        cleanup(false);
        setup(path, "weblogs", "10000", "256", true);
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

// 10k Models end


// 20k Models


bool WebLog_Inference_20000_32_Load = true;

static void WebLog_Inference_20000_32(benchmark::State &state) {
    if (WebLog_Inference_20000_32_Load) {
        WebLog_Inference_20000_32_Load = false;
        cleanup(false);
        setup(path, "weblogs", "20000", "32", true);
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

bool WebLog_Inference_20000_64_Load = true;

static void WebLog_Inference_20000_64(benchmark::State &state) {
    if (WebLog_Inference_20000_64_Load) {
        WebLog_Inference_20000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "weblogs", "20000", "64", true);
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

bool Inference_20000_128_WebLog_Load = true;

static void WebLog_Inference_20000_128(benchmark::State &state) {
    if (Inference_20000_128_WebLog_Load) {
        Inference_20000_128_WebLog_Load = false;
        cleanup(false);
        setup(path, "weblogs", "20000", "128", true);
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

bool WebLog_Inference_20000_256_Load = true;

static void WebLog_Inference_20000_256(benchmark::State &state) {
    if (WebLog_Inference_20000_256_Load) {
        WebLog_Inference_20000_256_Load = false;
        cleanup(false);
        setup(path, "weblogs", "20000", "256", true);
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

// 20k Models end


// 50k Models


bool WebLog_Inference_50000_32_Load = true;

static void WebLog_Inference_50000_32(benchmark::State &state) {
    if (WebLog_Inference_50000_32_Load) {
        WebLog_Inference_50000_32_Load = false;
        cleanup(false);
        setup(path, "weblogs", "50000", "32", true);
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

bool WebLog_Inference_50000_64_Load = true;

static void WebLog_Inference_50000_64(benchmark::State &state) {
    if (WebLog_Inference_50000_64_Load) {
        WebLog_Inference_50000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "weblogs", "50000", "64", true);
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

bool Inference_50000_128_WebLog_Load = true;

static void WebLog_Inference_50000_128(benchmark::State &state) {
    if (Inference_50000_128_WebLog_Load) {
        Inference_50000_128_WebLog_Load = false;
        cleanup(false);
        setup(path, "weblogs", "50000", "128", true);
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

bool WebLog_Inference_50000_256_Load = true;

static void WebLog_Inference_50000_256(benchmark::State &state) {
    if (WebLog_Inference_50000_256_Load) {
        WebLog_Inference_50000_256_Load = false;
        cleanup(false);
        setup(path, "weblogs", "50000", "256", true);
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

// 50k Models end


// 100k Models start

bool WebLog_Inference_100000_32_Load = true;

static void WebLog_Inference_100000_32(benchmark::State &state) {
    if (WebLog_Inference_100000_32_Load) {
        WebLog_Inference_100000_32_Load = false;
        cleanup(false);
        setup(path, "weblogs", "100000", "32", true);
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

bool WebLog_Inference_100000_64_Load = true;

static void WebLog_Inference_100000_64(benchmark::State &state) {
    if (WebLog_Inference_100000_64_Load) {
        WebLog_Inference_100000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "weblogs", "100000", "64", true);
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

bool Inference_100000_128_WebLog_Load = true;

static void WebLog_Inference_100000_128(benchmark::State &state) {
    if (Inference_100000_128_WebLog_Load) {
        Inference_100000_128_WebLog_Load = false;
        cleanup(false);
        setup(path, "weblogs", "100000", "128", true);
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

bool WebLog_Inference_100000_256_Load = true;

static void WebLog_Inference_100000_256(benchmark::State &state) {
    if (WebLog_Inference_100000_256_Load) {
        WebLog_Inference_100000_256_Load = false;
        cleanup(false);
        setup(path, "weblogs", "100000", "256", true);
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

// 100k Models start

static bool Weblog_Btree_128_LOAD = true;
tree_type btree_weblog;

static void Weblog_Btree_128(benchmark::State &state) {
    if (Weblog_Btree_128_LOAD) {
        Weblog_Btree_128_LOAD = false;
        cleanup(false);
        btree_insert(btree_weblog, tData, indices, 0, tData.size() - 1);
//        cout << "btree lognormal size: " << btree_weblog.size() << endl;
    }

    uint32_t sz = tData.size() - 1;
    uint32_t i = 0;
    for (auto _ : state) {
        escape(&sum);
        i = GetRandKey(sz);
        sum += btree_find(btree_weblog, tData[i]);
        clobber();
    }
}


static bool Inference_100000_128_Maps_Load = true;

static void Maps_Inference_100000_128(benchmark::State &state) {
    if (Inference_100000_128_Maps_Load) {
        Inference_100000_128_Maps_Load = false;
        cleanup(true);
        setup(path, "maps", "100000", "128", false);
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
        cleanup(true);
        setup(path, "lognormal", "100000", "128", false);
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
        cleanup(true);
        setup(path, "maps", "100000", "128", false);
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
        cleanup(true);
        setup(path, "lognormal", "100000", "128", false);
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

BENCHMARK(WebLog_Inference_10000_32);
BENCHMARK(WebLog_Inference_10000_64);
BENCHMARK(WebLog_Inference_10000_128);
BENCHMARK(WebLog_Inference_10000_256);

BENCHMARK(WebLog_Inference_20000_32);
BENCHMARK(WebLog_Inference_20000_64);
BENCHMARK(WebLog_Inference_20000_128);
BENCHMARK(WebLog_Inference_20000_256);

BENCHMARK(WebLog_Inference_50000_32);
BENCHMARK(WebLog_Inference_50000_64);
BENCHMARK(WebLog_Inference_50000_128);
BENCHMARK(WebLog_Inference_50000_256);

BENCHMARK(WebLog_Inference_100000_32);
BENCHMARK(WebLog_Inference_100000_64);
BENCHMARK(WebLog_Inference_100000_128);
BENCHMARK(WebLog_Inference_100000_256);

BENCHMARK(Weblog_Btree_128);

//BENCHMARK(Maps_Inference_100000_128);

//BENCHMARK(Lognormal_Inference_100000_128);

//BENCHMARK(MAPS_BTREE_128);
//BENCHMARK(LOGNORMAL_BTREE_128);

BENCHMARK_MAIN();