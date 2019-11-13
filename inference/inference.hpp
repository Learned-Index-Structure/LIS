#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include "iaca_mac/iacaMarks.h"

union Mat1x1 { //Just for benchmarking
    float m[1][1];
};

union Mat32x1 { //Just for benchmarking
    float m[32][1];
};

union Mat1x32 {
    float m[1][32];
    __m256 row[4];
};

union Mat32x32 {
    float m[32][32];
    __m256 row[32][4];
};


union Mat1x32d {
    double m[1][32];
    __m256d row[8];
};

union Mat32x32d {
    double m[32][32];
    __m256d row[32][8];
};


float val = 1;

inline static float randf() {
    // assumes VC++ rand()
    return (rand() - 16374.0f) / 1024.0f;
//    return val;
}

template<size_t N, size_t M, typename MatType>
inline static void randmat(MatType &mat) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            mat.m[i][j] = randf();
}

template<size_t N, size_t M, typename O, typename I1, typename I2>
inline void matmult_ref(O &out, const I1 &A, const I2 &B) {
    O t = {0}; // write to temp
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            for (int k = 0; k < 32; k++)
                t.m[i][j] += A.m[i][k] * B.m[k][j];
    out = t;
}


// template<size_t models, size_t rows, size_t cols>
// inline void load_layer_data(float (&layer_data)[models][rows][cols]) {
//     for (int i = 0; i < models; i++) {
//         for (int j = 0; j < rows; j++) {
//             for (int k = 0; k < cols; k++) {
//                 layer_data[i][j][k] = rand() % 10 + 1.0;
//             }
//         }
//     }
// }


// Mat1x1 naive_output;

// Mat1x32 key;
// //MM from first hidden layer output
// Mat1x32 out_1;
// Mat1x32 out_2;

// Mat1x32 hidden_layer_1;
// Mat32x32 hidden_layer_2;
// Mat1x32 output_layer;
// float lr_wt[2] = {3.11, 1.32};
// float layer_data[models][32][1] = {{{0}}};

// inline void LoadData() {
//     randmat<1, 1, Mat1x1>(naive_output);
//     randmat<1, 32, Mat1x32>(key);
//     randmat<1, 32, Mat1x32>(hidden_layer_1);
//     randmat<32, 32, Mat32x32>(hidden_layer_2);
//     randmat<1, 32, Mat1x32>(output_layer);
//     load_layer_data(layer_data);
// }

inline void matmult_AVX_1x32x32(Mat1x32 &out, const Mat1x32 &A, const Mat32x32 &B) {
    _mm256_zeroupper();
//    IACA_START
    __m256 A_vec = _mm256_broadcast_ss(&A.m[0][0]);
    __m256 result0 = _mm256_mul_ps(A_vec, B.row[0][0]);
    __m256 result1 = _mm256_mul_ps(A_vec, B.row[0][1]);
    __m256 result2 = _mm256_mul_ps(A_vec, B.row[0][2]);
    __m256 result3 = _mm256_mul_ps(A_vec, B.row[0][3]);

    A_vec = _mm256_broadcast_ss(&A.m[0][1]);
    result0 = _mm256_add_ps(result0, _mm256_mul_ps(A_vec, B.row[1][0]));
    result1 = _mm256_add_ps(result1, _mm256_mul_ps(A_vec, B.row[1][1]));
    result2 = _mm256_add_ps(result2, _mm256_mul_ps(A_vec, B.row[1][2]));
    result3 = _mm256_add_ps(result3, _mm256_mul_ps(A_vec, B.row[1][3]));

    A_vec = _mm256_broadcast_ss(&A.m[0][2]);
    result0 = _mm256_add_ps(result0, _mm256_mul_ps(A_vec, B.row[2][0]));
    result1 = _mm256_add_ps(result1, _mm256_mul_ps(A_vec, B.row[2][1]));
    result2 = _mm256_add_ps(result2, _mm256_mul_ps(A_vec, B.row[2][2]));
    result3 = _mm256_add_ps(result3, _mm256_mul_ps(A_vec, B.row[2][3]));

    A_vec = _mm256_broadcast_ss(&A.m[0][3]);
    result0 = _mm256_add_ps(result0, _mm256_mul_ps(A_vec, B.row[3][0]));
    result1 = _mm256_add_ps(result1, _mm256_mul_ps(A_vec, B.row[3][1]));
    result2 = _mm256_add_ps(result2, _mm256_mul_ps(A_vec, B.row[3][2]));
    result3 = _mm256_add_ps(result3, _mm256_mul_ps(A_vec, B.row[3][3]));

    for (int i = 4; i < 32; i += 4) {

        A_vec = _mm256_broadcast_ss(&A.m[0][i]);
        result0 = _mm256_add_ps(result0, _mm256_mul_ps(A_vec, B.row[i][0]));
        result1 = _mm256_add_ps(result1, _mm256_mul_ps(A_vec, B.row[i][1]));
        result2 = _mm256_add_ps(result2, _mm256_mul_ps(A_vec, B.row[i][2]));
        result3 = _mm256_add_ps(result3, _mm256_mul_ps(A_vec, B.row[i][3]));

        A_vec = _mm256_broadcast_ss(&A.m[0][i + 1]);
        result0 = _mm256_add_ps(result0, _mm256_mul_ps(A_vec, B.row[i + 1][0]));
        result1 = _mm256_add_ps(result1, _mm256_mul_ps(A_vec, B.row[i + 1][1]));
        result2 = _mm256_add_ps(result2, _mm256_mul_ps(A_vec, B.row[i + 1][2]));
        result3 = _mm256_add_ps(result3, _mm256_mul_ps(A_vec, B.row[i + 1][3]));

        A_vec = _mm256_broadcast_ss(&A.m[0][i + 2]);
        result0 = _mm256_add_ps(result0, _mm256_mul_ps(A_vec, B.row[i + 2][0]));
        result1 = _mm256_add_ps(result1, _mm256_mul_ps(A_vec, B.row[i + 2][1]));
        result2 = _mm256_add_ps(result2, _mm256_mul_ps(A_vec, B.row[i + 2][2]));
        result3 = _mm256_add_ps(result3, _mm256_mul_ps(A_vec, B.row[i + 2][3]));

        A_vec = _mm256_broadcast_ss(&A.m[0][i + 3]);
        result0 = _mm256_add_ps(result0, _mm256_mul_ps(A_vec, B.row[i + 3][0]));
        result1 = _mm256_add_ps(result1, _mm256_mul_ps(A_vec, B.row[i + 3][1]));
        result2 = _mm256_add_ps(result2, _mm256_mul_ps(A_vec, B.row[i + 3][2]));
        result3 = _mm256_add_ps(result3, _mm256_mul_ps(A_vec, B.row[i + 3][3]));

    }

    out.row[0] = result0;
    out.row[1] = result1;
    out.row[2] = result2;
    out.row[3] = result3;
}


inline void matmult_AVX_1x1x32(Mat1x32 &out, const Mat1x32 &A, const Mat1x32 &B) {
    __m256 m_vect = _mm256_broadcast_ss(&A.m[0][0]);
    out.row[0] = _mm256_mul_ps(m_vect, B.row[0]);
    out.row[1] = _mm256_mul_ps(m_vect, B.row[1]);
    out.row[2] = _mm256_mul_ps(m_vect, B.row[2]);
    out.row[3] = _mm256_mul_ps(m_vect, B.row[3]);
}

inline float matmult_AVX_1x32x1_REF(const Mat1x32 &A, const Mat1x32 &B) {
    __m256 result0 = {0.0};
    result0 = _mm256_add_ps(result0, _mm256_mul_ps(A.row[0], B.row[0]));
    result0 = _mm256_add_ps(result0, _mm256_mul_ps(A.row[1], B.row[1]));
    result0 = _mm256_add_ps(result0, _mm256_mul_ps(A.row[2], B.row[2]));
    result0 = _mm256_add_ps(result0, _mm256_mul_ps(A.row[3], B.row[3]));

    double result = 0.0f;
    for (int i = 0; i < 8; ++i) {
        result += result0[i];
    }
    return result;
}

inline float matmult_AVX_1x32x1(const Mat1x32 &A, const Mat1x32 &B) {
    // __m256 result0 = _mm256_mul_ps(_mm256_broadcast_ss(&A.m[0][0]), B.row[0]);
    // result0 = _mm256_add_ps(result0, _mm256_mul_ps(_mm256_broadcast_ss(&A.m[0][0]), B.row[1]));
    // result0 = _mm256_add_ps(result0, _mm256_mul_ps(_mm256_broadcast_ss(&A.m[0][0]), B.row[2]));
    // result0 = _mm256_add_ps(result0, _mm256_mul_ps(_mm256_broadcast_ss(&A.m[0][0]), B.row[3]));

    // // horizontal add of 8 16-bit partial sums and return result
    // result0 = _mm256_hadd_ps(result0, result0);
    // result0 = _mm256_hadd_ps(result0, result0);
    // result0 = _mm256_hadd_ps(result0, result0);
    float result = 0;
    for (int i = 0; i < 32; ++i) {
        result += A.m[0][i] * B.m[0][i];
    }

    return result;
}

template<typename T>
inline void Relu(T mat) {

}

template<size_t row, size_t mid, size_t col, typename O, typename I1, typename I2>
inline void MatMulNaive(O &out, const I1 &A, const I2 &B) {
    O t = {0};
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            for (int k = 0; k < mid; k++)
                t.m[i][j] += A.m[i][k] * B.m[k][j];

    out = t;
}

template<size_t m, size_t n, typename T>
inline void relu(T &out) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            out.m[i][j] = (out.m[i][j] > 0) ? out.m[i][j] : 0;
        }
    }
}

// inline float NaiveInference() {
//     MatMulNaive<1, 1, 32>(out_1, key, hidden_layer_1);
//     relu<1, 32>(out_1);
//     MatMulNaive<1, 32, 32>(out_2, out_1, hidden_layer_2);
//     relu<1, 32>(out_2);
//     MatMulNaive<1, 32, 1>(naive_output, out_2, output_layer);


//     float pred = naive_output.m[0][0] * models / N;
//     return (pred * lr_wt[0] + lr_wt[1]);
// }

// inline float SIMDInference() {
//     float position = 0.0;

//     matmult_AVX_1x1x32(out_1, key, hidden_layer_1);
//     relu<1, 32>(out_1);
//     matmult_AVX_1x32x32(out_2, out_1, hidden_layer_2);
//     relu<1, 32>(out_2);
//     position = matmult_AVX_1x32x1(out_2, output_layer);

//     float pred = position * models / N;
//     return (pred * lr_wt[0] + lr_wt[1]);
// }



double doubleVal = 1.0232;

inline static float randd() {
    if(doubleVal > 10000)
        doubleVal = 1.4532;
    return doubleVal++;
}


template<size_t N, size_t M, typename MatType>
inline static void randmatd(MatType &mat) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            mat.m[i][j] = randd();
}


inline void matmult_AVX_1x32x32d(Mat1x32d &out, const Mat1x32d &A, const Mat32x32d &B) {
    _mm256_zeroupper();
    __m256d result0 = _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][0]), B.row[0][0]);
    __m256d result1 = _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][0]), B.row[0][1]);
    __m256d result2 = _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][0]), B.row[0][2]);
    __m256d result3 = _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][0]), B.row[0][3]);
    __m256d result4 = _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][0]), B.row[0][4]);
    __m256d result5 = _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][0]), B.row[0][5]);
    __m256d result6 = _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][0]), B.row[0][6]);
    __m256d result7 = _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][0]), B.row[0][7]);

    for (int i = 1; i < 32; i++) {
        result0 = _mm256_add_pd(result0, _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][i]), B.row[i][0]));
        result1 = _mm256_add_pd(result1, _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][i]), B.row[i][1]));
        result2 = _mm256_add_pd(result2, _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][i]), B.row[i][2]));
        result3 = _mm256_add_pd(result3, _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][i]), B.row[i][3]));
        result4 = _mm256_add_pd(result4, _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][i]), B.row[i][4]));
        result5 = _mm256_add_pd(result5, _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][i]), B.row[i][5]));
        result6 = _mm256_add_pd(result6, _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][i]), B.row[i][6]));
        result7 = _mm256_add_pd(result7, _mm256_mul_pd(_mm256_broadcast_sd(&A.m[0][i]), B.row[i][7]));
    }

    out.row[0] = result0;
    out.row[1] = result1;
    out.row[2] = result2;
    out.row[3] = result3;
    out.row[4] = result4;
    out.row[5] = result5;
    out.row[6] = result6;
    out.row[7] = result7;
}


inline void matmult_AVX_1x1x32d(Mat1x32d &out, const Mat1x32d &A, const Mat1x32d &B) {
    __m256d m_vect = _mm256_broadcast_sd(&A.m[0][0]);
    out.row[0] = _mm256_mul_pd(m_vect, B.row[0]);
    out.row[1] = _mm256_mul_pd(m_vect, B.row[1]);
    out.row[2] = _mm256_mul_pd(m_vect, B.row[2]);
    out.row[3] = _mm256_mul_pd(m_vect, B.row[3]);
    out.row[4] = _mm256_mul_pd(m_vect, B.row[4]);
    out.row[5] = _mm256_mul_pd(m_vect, B.row[5]);
    out.row[6] = _mm256_mul_pd(m_vect, B.row[6]);
    out.row[7] = _mm256_mul_pd(m_vect, B.row[7]);
}

inline double matmult_AVX_1x32x1_REFd(const Mat1x32d &A, const Mat1x32d &B) {
    __m256d result0 = {0.0};

    result0 = _mm256_add_pd(result0, _mm256_mul_pd(A.row[0], B.row[0]));
    result0 = _mm256_add_pd(result0, _mm256_mul_pd(A.row[1], B.row[1]));
    result0 = _mm256_add_pd(result0, _mm256_mul_pd(A.row[2], B.row[2]));
    result0 = _mm256_add_pd(result0, _mm256_mul_pd(A.row[3], B.row[3]));
    result0 = _mm256_add_pd(result0, _mm256_mul_pd(A.row[4], B.row[4]));
    result0 = _mm256_add_pd(result0, _mm256_mul_pd(A.row[5], B.row[5]));
    result0 = _mm256_add_pd(result0, _mm256_mul_pd(A.row[6], B.row[6]));
    result0 = _mm256_add_pd(result0, _mm256_mul_pd(A.row[7], B.row[7]));

    double result = 0.0f;
    for (int i = 0; i < 4; ++i) {
        result += result0[i];
    }
    return result;
}
