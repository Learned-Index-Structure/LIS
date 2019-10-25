#ifndef MAT_H_
#define MAT_H_

#include <immintrin.h>

union Mat1x32 {
    float m[1][32];
    __m256 row[4];
};

union Mat1x32i {
    int m[1][32];
    __m256i row[4];
};

union Mat32x32 {
    float m[32][32];
    __m256 row[32][4];
};

union Mat32x32i {
    int m[32][32];
    __m256i row[32][4];
};

#endif