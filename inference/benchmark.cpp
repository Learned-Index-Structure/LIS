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

inline uint32_t GetRandKey(uint32_t max, uint32_t min) {
    return (uint32_t) ((rand() % (max - min + 1)) + min);
}

long long sum = 0;

static void Btree200M_RandKey(benchmark::State &state) {
    btree_insert(200000000, 400000000);
    int res = 0;
    for (auto _ : state) {
//        IACA_START
        escape(&tree);
        escape(&res);
        escape(&sum);
        res = btree_find(GetRandKey(400000000, 200000000));
        sum += res;
        clobber();
//        IACA_END
    }
//    std::cout << sum << std::endl;
}

uint32_t btreeKey = 323456789;


static void Btree200M_SameKey(benchmark::State &state) {
//    tree.clear();
//    btree_insert(200000000, 400000000);
    sum = 0;
    uint32_t res = 0;
    for (auto _ : state) {
        escape(&tree);
        escape(&res);
        escape(&sum);
        escape(&btreeKey);
        res = btree_find(btreeKey);
        sum += res;
        clobber();
    }
//    std::cout << sum << std::endl;
}

static void Btree200M_KeyPlusOne(benchmark::State &state) {
//    tree.clear();
//    btree_insert(200000000, 400000000);
    sum = 0;
    uint32_t res = 0;
    btreeKey = 323456789;
    for (auto _ : state) {
        escape(&tree);
        escape(&res);
        escape(&sum);
        escape(&btreeKey);
        res = btree_find((++btreeKey) % 400000000);
        sum += res;
        clobber();
    }
//    std::cout << sum << std::endl;
}


int bkey = 5;

static void Btree100_RandKey(benchmark::State &state) {
    tree.clear();
    btree_insert(434, 534);
    sum = 0;
    int res = 0;
    for (auto _ : state) {
        escape(&tree);
        escape(&res);
        res = btree_find(GetRandKey(534, 434));
        sum += (long long) res;
        clobber();
    }
//    std::cout << sum << std::endl;
}

static void Btree100_SameKey(benchmark::State &state) {
    sum = 0;
    int res = 0;
    bkey = 4;
    for (auto _ : state) {
        escape(&tree);
        escape(&res);
        escape(&sum);
        res = btree_find((bkey++) % 100);
        sum += res;
        clobber();
    }
//    std::cout << sum << std::endl;
}


static void Naive_MM_1x1x32(benchmark::State &state) {
    Mat1x1 A;
    Mat1x32 B, out;
    randmat<1, 1>(A);
    randmat<1, 32>(B);

    for (auto _ : state) {
        escape(&out);
        MatMulNaive<1, 1, 32>(out, A, B);
        clobber();
    }
}

static void Naive_MM_1x32x1(benchmark::State &state) {
    Mat1x32 A;
    Mat32x1 B;
    Mat1x1 out;
    randmat<1, 32>(A);
    randmat<32, 1>(B);

    for (auto _ : state) {
        escape(&out);
        MatMulNaive<1, 32, 1>(out, A, B);
        clobber();
    }
}

static void Naive_MM_1x32x32(benchmark::State &state) {
    Mat1x32 A, out;
    Mat32x32 B;
    randmat<1, 32>(A);
    randmat<32, 32>(B);

    for (auto _ : state) {
        escape(&out);
        MatMulNaive<1, 32, 32>(out, A, B);
        clobber();
    }
}

static void Naive_Inference(benchmark::State &state) {
    LoadData();
    for (auto _ : state) {
        float pred = 0;
        escape(&pred);
        pred = NaiveInference();
        clobber();
    }
}

static void SIMD_MM_1x1x32(benchmark::State &state) {
    Mat1x32 A, B, out;
    randmat<1, 32>(A);
    randmat<1, 32>(B);

    for (auto _ : state) {
        escape(&out);
        matmult_AVX_1x1x32(out, A, B);
        clobber();
    }
}

static void SIMD_MM_1x32x1(benchmark::State &state) {
    Mat1x32 A, B;
    randmat<1, 32>(A);
    randmat<1, 32>(B);

    for (auto _ : state) {
        float pred = 0;
        escape(&pred);
        pred = matmult_AVX_1x32x1(A, B);
        clobber();
    }
}


static void SIMD_MM_1x32x32(benchmark::State &state) {
    Mat1x32 A, out;
    Mat32x32 B;
    randmat<1, 32>(A);
    randmat<32, 32>(B);

    for (auto _ : state) {
        escape(&out);
        matmult_AVX_1x32x32(out, A, B);
        clobber();
    }
}

static void SIMD_Inference(benchmark::State &state) {
    LoadData();
    float pred = 0;
    for (auto _ : state) {
        escape(&pred);
        pred = SIMDInference();
        clobber();
    }
}

static void RandKeyGen(benchmark::State &state) {
    sum = 0;
    uint32_t pred = 0;
    for (auto _ : state) {
        escape(&pred);
        escape(&sum);
        pred = GetRandKey(420000000, 120000000);
        sum += pred;
        clobber();
    }
//    std::cout << sum << std::endl;
}

static void BinarySearch200M_RandKey(benchmark::State &state) {
    uint32_t n = 200000000;
    vec.clear();
    BinaryInsert(n);
    bkey = 563;
    sum = 0;
    uint32_t pred = 0;
    for (auto _ : state) {
        escape(&bkey);
        escape(&pred);
        escape(&sum);
        pred = BinarySearch(vec, GetRandKey(n, bkey), n);
        sum += pred;
        clobber();
    }
//    std::cout << sum << std::endl;
}

static void BinarySearch200M_KeyPlusOne(benchmark::State &state) {
    uint32_t n = 200000000;
    bkey = 30300;
    sum = 0;
    uint32_t pred = 0;
    for (auto _ : state) {
        escape(&bkey);
        escape(&pred);
        escape(&sum);
        pred = BinarySearch(vec, (bkey++) % n, n);
        sum += pred;
        clobber();
    }
//    std::cout << sum << std::endl;
}

static void BinarySearch100_RandKey(benchmark::State &state) {
    vec.clear();
    uint32_t n = 100;
    bkey = 5;
    sum = 0;
    uint32_t pred = 0;
    BinaryInsert(n);
    for (auto _ : state) {
        escape(&pred);
        escape(&bkey);
        escape(&sum);
        pred = BinarySearch(vec, GetRandKey(100, 1), n);
        sum += pred;
        clobber();
    }
//    std::cout << sum << std::endl;
}

static void BinarySearch100_SameKey(benchmark::State &state) {
    vec.clear();
    uint32_t n = 100;
    bkey = 5;
    sum = 0;
    uint32_t pred = 0;
    for (auto _ : state) {
        escape(&bkey);
        escape(&pred);
        escape(&sum);
        pred = BinarySearch(vec, (bkey++) % n, n);
        sum += pred;
        clobber();
    }
//    std::cout << sum << std::endl;
}


BENCHMARK(Naive_MM_1x1x32);
BENCHMARK(Naive_MM_1x32x1);
BENCHMARK(Naive_MM_1x32x32);
BENCHMARK(Naive_Inference);
BENCHMARK(SIMD_MM_1x1x32);
BENCHMARK(SIMD_MM_1x32x1);
BENCHMARK(SIMD_MM_1x32x32);
BENCHMARK(SIMD_Inference);

BENCHMARK(RandKeyGen);
BENCHMARK(Btree200M_RandKey);
BENCHMARK(Btree200M_SameKey);
BENCHMARK(Btree200M_KeyPlusOne);
BENCHMARK(Btree100_RandKey);
BENCHMARK(Btree100_SameKey);

BENCHMARK(BinarySearch200M_RandKey);
BENCHMARK(BinarySearch200M_KeyPlusOne);
BENCHMARK(BinarySearch100_RandKey);
BENCHMARK(BinarySearch100_SameKey);

BENCHMARK_MAIN();