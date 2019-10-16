// g++ -O3 -mavx2 -lpthread -std=c++11 -isystem benchmark/include -Lbenchmark/build/src -lbenchmark -lpthread -o bench benchmark.cpp
// ./bench

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

inline int GetRandKey(int val) {
    return (int) (rand() % 100 + val);
}


int mask = 1;
uint32_t btreeKey = 123456789;

static void Btree200M_SameKey(benchmark::State &state) {
    btree_insert(200000000, 400000000);

    for (auto _ : state) {
        //IACA_START
        int key = btreeKey & mask;
        escape(&key);
        int res = btree_find(key);
        //IACA_END
        escape(&tree);
        escape(&res);
    }
}

BENCHMARK(Btree200M_SameKey);

static void Btree200M_RandKey(benchmark::State &state) {
    for (auto _ : state) {
        int res = btree_find(GetRandKey(302003000));
        escape(&tree);
        escape(&res);
    }
}

BENCHMARK(Btree200M_RandKey);


int bkey = 3555;

static void Btree1000_SameKey(benchmark::State &state) {
    tree.clear();
    btree_insert(3434, 4334);

    for (auto _ : state) {
        btree_find(bkey);
        escape(&tree);
        escape(&val);
    }
}

BENCHMARK(Btree1000_SameKey);

static void Btree1000_RandKey(benchmark::State &state) {
    for (auto _ : state) {
        btree_find(GetRandKey(bkey));
        escape(&tree);
        escape(&val);
        clobber();
    }
}

BENCHMARK(Btree1000_RandKey);

static void MM_Naive_1x1x32(benchmark::State &state) {
    Mat1x1 A;
    Mat1x32 B, out;
    randmat<1, 1>(A);
    randmat<1, 32>(B);

    for (auto _ : state) {
        MatMulNaive<1, 1, 32>(out, A, B);
        escape(&out);
    }
}

BENCHMARK(MM_Naive_1x1x32);

static void MM_Naive_1x32x1(benchmark::State &state) {
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

BENCHMARK(MM_Naive_1x32x1);

static void MM_Naive_1x32x32(benchmark::State &state) {
    Mat1x32 A, out;
    Mat32x32 B;
    randmat<1, 32>(A);
    randmat<32, 32>(B);

    for (auto _ : state) {
        MatMulNaive<1, 32, 32>(out, A, B);
        escape(&out);
    }
}

BENCHMARK(MM_Naive_1x32x32);

static void Naive_Inference(benchmark::State &state) {
    LoadData();
    for (auto _ : state) {
        float pred = NaiveInference();
        escape(&pred);
    }
}

BENCHMARK(Naive_Inference);

static void SIMD_MM_1x1x32(benchmark::State &state) {
    Mat1x32 A, B, out;
    randmat<1, 32>(A);
    randmat<1, 32>(B);

    for (auto _ : state) {
        matmult_AVX_1x1x32(out, A, B);
        escape(&out);
    }
}

BENCHMARK(SIMD_MM_1x1x32);

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

BENCHMARK(SIMD_MM_1x32x1);

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

BENCHMARK(SIMD_MM_1x32x32);

static void SIMD_Inference(benchmark::State &state) {
    LoadData();
    for (auto _ : state) {
        float pred = SIMDInference();
        escape(&pred);
    }
}

BENCHMARK(SIMD_Inference);

static void RandKeyGen(benchmark::State &state) {
    for (auto _ : state) {
        uint32_t pred = GetRandKey(bkey + 300000000);
        escape(&pred);
    }
}

BENCHMARK(RandKeyGen);

static void BinarySearch200M(benchmark::State &state) {
    uint32_t n = 200000000;
    BinaryInsert(n);
    for (auto _ : state) {
        //IACA_START
        uint32_t pred = BinarySearch(vec, GetRandKey(bkey + 300000000), n);
        //IACA_END
        escape(&pred);
        escape(&vec);
    }
}

BENCHMARK(BinarySearch200M);
int i = 2;

static void BinarySearch500(benchmark::State &state) {
    vec.clear();
    uint32_t n = 100;
    BinaryInsert(n);
    for (auto _ : state) {
        uint32_t pred = BinarySearch(vec, 48 + (i++), n);
        escape(&pred);
        clobber();
    }
}

BENCHMARK(BinarySearch500);

BENCHMARK_MAIN();