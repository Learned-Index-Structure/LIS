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

uint32_t randInd = 12;

inline uint32_t GetRandKey(uint32_t sz) {
    return (uint32_t) ((rand()) % sz);
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
    const int size = static_cast<int>(state.range(0));
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

//    state.SetItemsProcessed(state.iterations() * size);
//    state.SetBytesProcessed(state.iterations() * size);
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
    const int size = static_cast<int>(state.range(0));

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
//    state.SetItemsProcessed(state.iterations() * size);
//    state.SetBytesProcessed(state.iterations() * size);
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

//
//static bool Inference_100000_128_Maps_Load = true;
//
//static void Maps_Inference_100000_128(benchmark::State &state) {
//    if (Inference_100000_128_Maps_Load) {
//        Inference_100000_128_Maps_Load = false;
//        cleanup(true);
//        setup(path, "maps", "100000", "128", false);
//        getKeyList(tData, dataLines, maxKey);
//    }
//    double keyToSearch;
//    uint32_t sz = keyList.size() - 1;
//    uint32_t i = 0;
//    for (auto _ : state) {
//        escape(&sum);
//        i = GetRandKey(sz);
//        keyToSearch = keyList[i];
//        key.m[0][0] = keyToSearch;
//        sum += infer(keyListInt[i]);
//        clobber();
//    }
//}
//
//static bool Inference_100000_128_Lognormal_Load = true;
//
//static void Lognormal_Inference_100000_128(benchmark::State &state) {
//    if (Inference_100000_128_Lognormal_Load) {
//        Inference_100000_128_Lognormal_Load = false;
//        cleanup(true);
//        setup(path, "lognormal", "100000", "128", false);
//        getKeyList(tData, dataLines, maxKey);
//    }
//    double keyToSearch;
//    uint32_t sz = keyList.size() - 1;
//    uint32_t i = 0;
//    for (auto _ : state) {
//        escape(&sum);
//        i = GetRandKey(sz);
//        keyToSearch = keyList[i];
//        key.m[0][0] = keyToSearch;
//        sum += infer(keyListInt[i]);
//        clobber();
//    }
//}
//
//static bool MAPS_BTREE_128_LOAD = true;
//tree_type btree_maps;
//
//static void MAPS_BTREE_128(benchmark::State &state) {
//    if (MAPS_BTREE_128_LOAD) {
//        MAPS_BTREE_128_LOAD = false;
//        cleanup(true);
//        setup(path, "maps", "100000", "128", false);
//        cout << tData.size() << " " << indices.size() << endl;
//        btree_insert(btree_maps, tData, indices, 0, tData.size() - 1);
//        cout << tData.size() << " " << indices.size() << endl;
//    }
//
//    uint32_t sz = tData.size() - 1;
//    uint32_t i = 0;
//    for (auto _ : state) {
//        escape(&sum);
//        i = GetRandKey(sz);
//        sum += btree_find(btree_maps, tData[i]);
//        clobber();
//    }
//}
//
//static bool LOGNORMAL_BTREE_128_LOAD = true;
//tree_type btree_lognormal;
//
//static void LOGNORMAL_BTREE_128(benchmark::State &state) {
//    if (LOGNORMAL_BTREE_128_LOAD) {
//        LOGNORMAL_BTREE_128_LOAD = false;
//        cleanup(true);
//        setup(path, "lognormal", "100000", "128", false);
//        btree_insert(btree_lognormal, tData, indices, 0, tData.size() - 1);
//        cout << "btree lognormal size: " << btree_lognormal.size() << endl;
//    }
//
//    uint32_t sz = tData.size() - 1;
//    uint32_t i = 0;
//    for (auto _ : state) {
//        escape(&sum);
//        i = GetRandKey(sz);
//        sum += btree_find(btree_lognormal, tData[i]);
//        clobber();
//    }
//}



bool Maps_Inference_10000_32_Load = true;

static void Maps_Inference_10000_32(benchmark::State &state) {
    if (Maps_Inference_10000_32_Load) {
        Maps_Inference_10000_32_Load = false;
        cleanup(true);
        btree_weblog.clear();
        setup(path, "maps", "10000", "32", false);
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

bool Maps_Inference_10000_64_Load = true;

static void Maps_Inference_10000_64(benchmark::State &state) {
    if (Maps_Inference_10000_64_Load) {
        Maps_Inference_10000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "maps", "10000", "64", true);
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

bool Inference_10000_128_Maps_Load = true;

static void Maps_Inference_10000_128(benchmark::State &state) {
    if (Inference_10000_128_Maps_Load) {
        Inference_10000_128_Maps_Load = false;
        cleanup(false);
        setup(path, "maps", "10000", "128", true);
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

bool Maps_Inference_10000_256_Load = true;

static void Maps_Inference_10000_256(benchmark::State &state) {
    if (Maps_Inference_10000_256_Load) {
        Maps_Inference_10000_256_Load = false;
        cleanup(false);
        setup(path, "maps", "10000", "256", true);
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


bool Maps_Inference_20000_32_Load = true;

static void Maps_Inference_20000_32(benchmark::State &state) {
    if (Maps_Inference_20000_32_Load) {
        Maps_Inference_20000_32_Load = false;
        cleanup(false);
        setup(path, "maps", "20000", "32", true);
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

bool Maps_Inference_20000_64_Load = true;

static void Maps_Inference_20000_64(benchmark::State &state) {
    if (Maps_Inference_20000_64_Load) {
        Maps_Inference_20000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "maps", "20000", "64", true);
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

bool Inference_20000_128_Maps_Load = true;

static void Maps_Inference_20000_128(benchmark::State &state) {
    if (Inference_20000_128_Maps_Load) {
        Inference_20000_128_Maps_Load = false;
        cleanup(false);
        setup(path, "maps", "20000", "128", true);
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

bool Maps_Inference_20000_256_Load = true;

static void Maps_Inference_20000_256(benchmark::State &state) {
    if (Maps_Inference_20000_256_Load) {
        Maps_Inference_20000_256_Load = false;
        cleanup(false);
        setup(path, "maps", "20000", "256", true);
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


bool Maps_Inference_50000_32_Load = true;

static void Maps_Inference_50000_32(benchmark::State &state) {
    if (Maps_Inference_50000_32_Load) {
        Maps_Inference_50000_32_Load = false;
        cleanup(false);
        setup(path, "maps", "50000", "32", true);
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

bool Maps_Inference_50000_64_Load = true;

static void Maps_Inference_50000_64(benchmark::State &state) {
    if (Maps_Inference_50000_64_Load) {
        Maps_Inference_50000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "maps", "50000", "64", true);
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

bool Inference_50000_128_Maps_Load = true;

static void Maps_Inference_50000_128(benchmark::State &state) {
    if (Inference_50000_128_Maps_Load) {
        Inference_50000_128_Maps_Load = false;
        cleanup(false);
        setup(path, "maps", "50000", "128", true);
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

bool Maps_Inference_50000_256_Load = true;

static void Maps_Inference_50000_256(benchmark::State &state) {
    if (Maps_Inference_50000_256_Load) {
        Maps_Inference_50000_256_Load = false;
        cleanup(false);
        setup(path, "maps", "50000", "256", true);
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

bool Maps_Inference_100000_32_Load = true;

static void Maps_Inference_100000_32(benchmark::State &state) {
    if (Maps_Inference_100000_32_Load) {
        Maps_Inference_100000_32_Load = false;
        cleanup(false);
        setup(path, "maps", "100000", "32", true);
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

bool Maps_Inference_100000_64_Load = true;

static void Maps_Inference_100000_64(benchmark::State &state) {
    if (Maps_Inference_100000_64_Load) {
        Maps_Inference_100000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "maps", "100000", "64", true);
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

bool Inference_100000_128_Maps_Load = true;

static void Maps_Inference_100000_128(benchmark::State &state) {
    if (Inference_100000_128_Maps_Load) {
        Inference_100000_128_Maps_Load = false;
        cleanup(false);
        setup(path, "maps", "100000", "128", true);
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

bool Maps_Inference_100000_256_Load = true;

static void Maps_Inference_100000_256(benchmark::State &state) {
    if (Maps_Inference_100000_256_Load) {
        Maps_Inference_100000_256_Load = false;
        cleanup(false);
        setup(path, "maps", "100000", "256", true);
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

// 100k Models ends


bool Maps_Inference_200000_32_Load = true;

static void Maps_Inference_200000_32(benchmark::State &state) {
    if (Maps_Inference_200000_32_Load) {
        Maps_Inference_200000_32_Load = false;
        cleanup(false);
        setup(path, "maps", "200000", "32", true);
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

bool Maps_Inference_200000_64_Load = true;

static void Maps_Inference_200000_64(benchmark::State &state) {
    if (Maps_Inference_200000_64_Load) {
        Maps_Inference_200000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "maps", "200000", "64", true);
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

bool Inference_200000_128_Maps_Load = true;

static void Maps_Inference_200000_128(benchmark::State &state) {
    if (Inference_200000_128_Maps_Load) {
        Inference_200000_128_Maps_Load = false;
        cleanup(false);
        setup(path, "maps", "200000", "128", true);
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

bool Maps_Inference_200000_256_Load = true;

static void Maps_Inference_200000_256(benchmark::State &state) {
    if (Maps_Inference_200000_256_Load) {
        Maps_Inference_200000_256_Load = false;
        cleanup(false);
        setup(path, "maps", "200000", "256", true);
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

static bool Maps_Btree_128_LOAD = true;
tree_type btree_maps;

static void Maps_Btree_128(benchmark::State &state) {
    if (Maps_Btree_128_LOAD) {
        Maps_Btree_128_LOAD = false;
        cleanup(false);
        btree_insert(btree_maps, tData, indices, 0, tData.size() - 1);
//        cout << "Maps btree size: " << btree_maps.size() << endl;
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


bool Lognormal_Inference_10000_32_Load = true;

static void Lognormal_Inference_10000_32(benchmark::State &state) {
    if (Lognormal_Inference_10000_32_Load) {
        Lognormal_Inference_10000_32_Load = false;
        cleanup(true);
        btree_maps.clear();
        setup(path, "lognormal", "10000", "32", false);
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

bool Lognormal_Inference_10000_64_Load = true;

static void Lognormal_Inference_10000_64(benchmark::State &state) {
    if (Lognormal_Inference_10000_64_Load) {
        Lognormal_Inference_10000_64_Load = false;
        cleanup(false);
        setup(path, "lognormal", "10000", "64", true);
        getKeyList(tData, dataLines, maxKey);
//        cout << "tData.size()  << dataLines << keyset: " << tData.size() << ", " << dataLines << ", " << keyList.size() << endl;
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

bool Inference_10000_128_Lognormal_Load = true;

static void Lognormal_Inference_10000_128(benchmark::State &state) {
    if (Inference_10000_128_Lognormal_Load) {
        Inference_10000_128_Lognormal_Load = false;
        cleanup(false);
        setup(path, "lognormal", "10000", "128", true);
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

bool Lognormal_Inference_10000_256_Load = true;

static void Lognormal_Inference_10000_256(benchmark::State &state) {
    if (Lognormal_Inference_10000_256_Load) {
        Lognormal_Inference_10000_256_Load = false;
        cleanup(false);
        setup(path, "lognormal", "10000", "256", true);
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


bool Lognormal_Inference_20000_32_Load = true;

static void Lognormal_Inference_20000_32(benchmark::State &state) {
    if (Lognormal_Inference_20000_32_Load) {
        Lognormal_Inference_20000_32_Load = false;
        cleanup(false);
        setup(path, "lognormal", "20000", "32", true);
        getKeyList(tData, dataLines, maxKey);
//        cout << "tData.size()  << dataLines << keyset: " << tData.size() << ", " << dataLines << ", " << keyList.size() << endl;
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

bool Lognormal_Inference_20000_64_Load = true;

static void Lognormal_Inference_20000_64(benchmark::State &state) {
    if (Lognormal_Inference_20000_64_Load) {
        Lognormal_Inference_20000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "lognormal", "20000", "64", true);
        getKeyList(tData, dataLines, maxKey);
//        cout << "tData.size()  << dataLines << keyset: " << tData.size() << ", " << dataLines << ", " << keyList.size() << endl;
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

bool Inference_20000_128_Lognormal_Load = true;

static void Lognormal_Inference_20000_128(benchmark::State &state) {
    if (Inference_20000_128_Lognormal_Load) {
        Inference_20000_128_Lognormal_Load = false;
        cleanup(false);
        setup(path, "lognormal", "20000", "128", true);
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

bool Lognormal_Inference_20000_256_Load = true;

static void Lognormal_Inference_20000_256(benchmark::State &state) {
    if (Lognormal_Inference_20000_256_Load) {
        Lognormal_Inference_20000_256_Load = false;
        cleanup(false);
        setup(path, "lognormal", "20000", "256", true);
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


bool Lognormal_Inference_50000_32_Load = true;

static void Lognormal_Inference_50000_32(benchmark::State &state) {
    if (Lognormal_Inference_50000_32_Load) {
        Lognormal_Inference_50000_32_Load = false;
        cleanup(false);
        setup(path, "lognormal", "50000", "32", true);
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

bool Lognormal_Inference_50000_64_Load = true;

static void Lognormal_Inference_50000_64(benchmark::State &state) {
    if (Lognormal_Inference_50000_64_Load) {
        Lognormal_Inference_50000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "lognormal", "50000", "64", true);
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

bool Inference_50000_128_Lognormal_Load = true;

static void Lognormal_Inference_50000_128(benchmark::State &state) {
    if (Inference_50000_128_Lognormal_Load) {
        Inference_50000_128_Lognormal_Load = false;
        cleanup(false);
        setup(path, "lognormal", "50000", "128", true);
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

bool Lognormal_Inference_50000_256_Load = true;

static void Lognormal_Inference_50000_256(benchmark::State &state) {
    if (Lognormal_Inference_50000_256_Load) {
        Lognormal_Inference_50000_256_Load = false;
        cleanup(false);
        setup(path, "lognormal", "50000", "256", true);
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

bool Lognormal_Inference_100000_32_Load = true;

static void Lognormal_Inference_100000_32(benchmark::State &state) {
    if (Lognormal_Inference_100000_32_Load) {
        Lognormal_Inference_100000_32_Load = false;
        cleanup(false);
        setup(path, "lognormal", "100000", "32", true);
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

bool Lognormal_Inference_100000_64_Load = true;

static void Lognormal_Inference_100000_64(benchmark::State &state) {
    if (Lognormal_Inference_100000_64_Load) {
        Lognormal_Inference_100000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "lognormal", "100000", "64", true);
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

bool Inference_100000_128_Lognormal_Load = true;

static void Lognormal_Inference_100000_128(benchmark::State &state) {
    if (Inference_100000_128_Lognormal_Load) {
        Inference_100000_128_Lognormal_Load = false;
        cleanup(false);
        setup(path, "lognormal", "100000", "128", true);
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

bool Lognormal_Inference_100000_256_Load = true;

static void Lognormal_Inference_100000_256(benchmark::State &state) {
    if (Lognormal_Inference_100000_256_Load) {
        Lognormal_Inference_100000_256_Load = false;
        cleanup(false);
        setup(path, "lognormal", "100000", "256", true);
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

// 100k Models ends

bool Lognormal_Inference_200000_32_Load = true;

static void Lognormal_Inference_200000_32(benchmark::State &state) {
    if (Lognormal_Inference_200000_32_Load) {
        Lognormal_Inference_200000_32_Load = false;
        cleanup(false);
        setup(path, "lognormal", "200000", "32", true);
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

bool Lognormal_Inference_200000_64_Load = true;

static void Lognormal_Inference_200000_64(benchmark::State &state) {
    if (Lognormal_Inference_200000_64_Load) {
        Lognormal_Inference_200000_64_Load = false;
        cleanup(false);
//        cout << "tData.size()  << dataLines << maxKey: " << tData.size() << ", " << dataLines << ", " << maxKey << endl;
        setup(path, "lognormal", "200000", "64", true);
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

bool Inference_200000_128_Lognormal_Load = true;

static void Lognormal_Inference_200000_128(benchmark::State &state) {
    if (Inference_200000_128_Lognormal_Load) {
        Inference_200000_128_Lognormal_Load = false;
        cleanup(false);
        setup(path, "lognormal", "200000", "128", true);
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

bool Lognormal_Inference_200000_256_Load = true;

static void Lognormal_Inference_200000_256(benchmark::State &state) {
    if (Lognormal_Inference_200000_256_Load) {
        Lognormal_Inference_200000_256_Load = false;
        cleanup(false);
        setup(path, "lognormal", "200000", "256", true);
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

static bool Lognormal_Btree_128_LOAD = true;
tree_type btree_lognormal;

static void Lognormal_Btree_128(benchmark::State &state) {
    if (Lognormal_Btree_128_LOAD) {
        Lognormal_Btree_128_LOAD = false;
        cleanup(false);
        btree_insert(btree_lognormal, tData, indices, 0, tData.size() - 1);
//        cout << "btree lognormal size: " << btree_lognormal.size() << endl;
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

BENCHMARK(WebLog_Inference_10000_32); //->Range(8, 8<<10);
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

BENCHMARK(WebLog_Inference_100000_32);//->Range(8, 8<<10);
BENCHMARK(WebLog_Inference_100000_64);
BENCHMARK(WebLog_Inference_100000_128);
BENCHMARK(WebLog_Inference_100000_256);

BENCHMARK(Weblog_Btree_128);

BENCHMARK(Maps_Inference_10000_32);
BENCHMARK(Maps_Inference_10000_64);
BENCHMARK(Maps_Inference_10000_128);
BENCHMARK(Maps_Inference_10000_256);

BENCHMARK(Maps_Inference_20000_32);
BENCHMARK(Maps_Inference_20000_64);
BENCHMARK(Maps_Inference_20000_128);
BENCHMARK(Maps_Inference_20000_256);

BENCHMARK(Maps_Inference_50000_32);
BENCHMARK(Maps_Inference_50000_64);
BENCHMARK(Maps_Inference_50000_128);
BENCHMARK(Maps_Inference_50000_256);

BENCHMARK(Maps_Inference_100000_32);
BENCHMARK(Maps_Inference_100000_64);
BENCHMARK(Maps_Inference_100000_128);
BENCHMARK(Maps_Inference_100000_256);

BENCHMARK(Maps_Inference_200000_32);
BENCHMARK(Maps_Inference_200000_64);
BENCHMARK(Maps_Inference_200000_128);
BENCHMARK(Maps_Inference_200000_256);

BENCHMARK(Maps_Btree_128);

BENCHMARK(Lognormal_Inference_10000_32);
BENCHMARK(Lognormal_Inference_10000_64);
BENCHMARK(Lognormal_Inference_10000_128);
BENCHMARK(Lognormal_Inference_10000_256);

BENCHMARK(Lognormal_Inference_20000_32);
BENCHMARK(Lognormal_Inference_20000_64);
BENCHMARK(Lognormal_Inference_20000_128);
BENCHMARK(Lognormal_Inference_20000_256);

BENCHMARK(Lognormal_Inference_50000_32);
BENCHMARK(Lognormal_Inference_50000_64);
BENCHMARK(Lognormal_Inference_50000_128);
BENCHMARK(Lognormal_Inference_50000_256);

BENCHMARK(Lognormal_Inference_100000_32);
BENCHMARK(Lognormal_Inference_100000_64);
BENCHMARK(Lognormal_Inference_100000_128);
BENCHMARK(Lognormal_Inference_100000_256);

BENCHMARK(Lognormal_Inference_200000_32);
BENCHMARK(Lognormal_Inference_200000_64);
BENCHMARK(Lognormal_Inference_200000_128);
BENCHMARK(Lognormal_Inference_200000_256);

BENCHMARK(Lognormal_Btree_128);

BENCHMARK_MAIN();
