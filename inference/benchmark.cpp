// g++ -O3 -mavx2 -lpthread -std=c++11 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o bench benchmark.cpp && ./bench
// g++-8 -o bench -O3 -mavx -pthread benchmark.cpp -Lbenchmark/build/src -lbenchmark && ./bench

#include "benchmark/include/benchmark/benchmark.h"

#include <iostream>
#include "btree.h"
#include "lms_algo.h"
#include "inference.h"

#include "iaca_mac/iacaMarks.h"


//Ref: https://www.youtube.com/watch?v=nXaxk27zwlk
static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void clobber() {
    asm volatile("" : : : "memory");
}

inline uint32_t GetRandKey(uint32_t max, uint32_t val) {
    return (uint32_t) ((rand() % max) + val);
}


uint32_t btreeKey = 123456789;

static void Btree200M_SameKey(benchmark::State &state) {
    for (auto _ : state) {
        //IACA_START

        escape(&key);
        int res = btree_find((btreeKey++));
        //IACA_END
        escape(&tree);
        escape(&res);
        escape(&btreeKey);
    }
}


static void Btree200M_RandKey(benchmark::State &state) {
    btree_insert(200000000, 400000000);
    for (auto _ : state) {
        int res = btree_find(GetRandKey(200000000, 302003000));
        escape(&tree);
        escape(&res);
    }
}

int bkey = 3555;

static void Btree1000_SameKey(benchmark::State &state) {
    for (auto _ : state) {
        btree_find(bkey);
        escape(&tree);
        escape(&val);
    }
}


static void Btree1000_RandKey(benchmark::State &state) {
    tree.clear();
    btree_insert(3434, 4334);

    for (auto _ : state) {
        btree_find(GetRandKey(3434, bkey));
        escape(&tree);
        escape(&val);
        clobber();
    }
}

static void Naive_MM_1x1x32(benchmark::State &state) {
    Mat1x1 A;
    Mat1x32 B, out;
    randmat<1, 1>(A);
    randmat<1, 32>(B);

    for (auto _ : state) {
        MatMulNaive<1, 1, 32>(out, A, B);
        escape(&out);
    }
}

static void Naive_MM_1x32x1(benchmark::State &state) {
    Mat1x32 A;
    Mat32x1 B;
    Mat1x1 out;
    randmat<1, 32>(A);
    randmat<32, 1>(B);

    for (auto _ : state) {
        MatMulNaive<1, 32, 1>(out, A, B);
        escape(&out);
    }
}

static void Naive_MM_1x32x32(benchmark::State &state) {
    Mat1x32 A, out;
    Mat32x32 B;
    randmat<1, 32>(A);
    randmat<32, 32>(B);

    for (auto _ : state) {
        MatMulNaive<1, 32, 32>(out, A, B);
        escape(&out);
    }
}

static void Naive_Inference(benchmark::State &state) {
    LoadData();
    for (auto _ : state) {
        float pred = NaiveInference();
        escape(&pred);
    }
}

static void SIMD_MM_1x1x32(benchmark::State &state) {
    Mat1x32 A, B, out;
    randmat<1, 32>(A);
    randmat<1, 32>(B);

    for (auto _ : state) {
        matmult_AVX_1x1x32(out, A, B);
        escape(&out);
    }
}

static void SIMD_MM_1x32x1(benchmark::State &state) {
    Mat1x32 A, B;
    randmat<1, 32>(A);
    randmat<1, 32>(B);

    for (auto _ : state) {
        float pred = matmult_AVX_1x32x1(A, B);
        escape(&pred);
        clobber();
    }
}


static void SIMD_MM_1x32x32(benchmark::State &state) {
    Mat1x32 A, out;
    Mat32x32 B;
    randmat<1, 32>(A);
    randmat<32, 32>(B);

    for (auto _ : state) {
        matmult_AVX_1x32x32(out, A, B);
        escape(&out);
    }
}

static void SIMD_Inference(benchmark::State &state) {
    LoadData();
    for (auto _ : state) {
        float pred = SIMDInference();
        escape(&pred);
    }
}

static void RandKeyGen(benchmark::State &state) {
    for (auto _ : state) {
        uint32_t pred = GetRandKey(100, (bkey++) + 300000);
        escape(&pred);
    }
}

static void BinarySearch200M(benchmark::State &state) {
    uint32_t n = 200000000;
    BinaryInsert(n);
    bkey = 223344;
    for (auto _ : state) {
        //IACA_START
        uint32_t pred = BinarySearch(vec, (GetRandKey(n, bkey) + 3000), n);
        //IACA_END
        escape(&pred);
    }
}


int i = 2;

static void BinarySearch500(benchmark::State &state) {
    vec.clear();
    uint32_t n = 100;
    bkey = 5;
    BinaryInsert(n);
    for (auto _ : state) {
        uint32_t pred = BinarySearch(vec, GetRandKey(n, bkey) + 3, n);
        escape(&pred);
        clobber();
    }
}

BENCHMARK(RandKeyGen);
BENCHMARK(Btree200M_RandKey);
BENCHMARK(Btree200M_SameKey);
BENCHMARK(Btree1000_RandKey);
BENCHMARK(Btree1000_SameKey);
BENCHMARK(Naive_MM_1x1x32);
BENCHMARK(Naive_MM_1x32x1);
BENCHMARK(Naive_MM_1x32x32);
BENCHMARK(Naive_Inference);
BENCHMARK(SIMD_MM_1x1x32);
BENCHMARK(SIMD_MM_1x32x1);
BENCHMARK(SIMD_MM_1x32x32);
BENCHMARK(SIMD_Inference);
BENCHMARK(BinarySearch200M);
BENCHMARK(BinarySearch500);

BENCHMARK_MAIN();