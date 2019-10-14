#include "avx_mm.h"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>


using namespace std;
typedef chrono::high_resolution_clock Clock;
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

float val = 1.0;

static float randf() {
    // assumes VC++ rand()
    //    return (rand() - 16384.0f) / 1024.0f;
    return val++;
}

template <size_t N, size_t M, typename MatType>
static void randmat(MatType &mat) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            mat.m[i][j] = randf();
}


static void randmat(Mat32x32 &M) {
    val = 1.0;
    for (int i = 0; i < 32; i++)
        for (int j = 0; j < 32; j++)
            M.m[i][j] = randf();
}

template <size_t N, size_t M, typename O, typename I1, typename I2>
void matmult_ref(O &out, const I1 &A, const I2 &B) {
    O t = {0.0}; // write to temp
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            for (int k = 0; k < 32; k++)
                t.m[i][j] += A.m[i][k] * B.m[k][j];

    out = t;
}

void matmult_AVX_1x32x32(Mat1x32 &out, const Mat1x32 &A, const Mat32x32 &B) {
    _mm256_zeroupper();
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

    for (int i = 4; i < 32; i+=4) {
        A_vec = _mm256_broadcast_ss(&A.m[0][i]);
        result0 = _mm256_add_ps(result0, _mm256_mul_ps(A_vec, B.row[i][0]));
        result1 = _mm256_add_ps(result1, _mm256_mul_ps(A_vec, B.row[i][1]));
        result2 = _mm256_add_ps(result2, _mm256_mul_ps(A_vec, B.row[i][2]));
        result3 = _mm256_add_ps(result3, _mm256_mul_ps(A_vec, B.row[i][3]));

        A_vec = _mm256_broadcast_ss(&A.m[0][i+1]);
        result0 = _mm256_add_ps(result0, _mm256_mul_ps(A_vec, B.row[i+1][0]));
        result1 = _mm256_add_ps(result1, _mm256_mul_ps(A_vec, B.row[i+1][1]));
        result2 = _mm256_add_ps(result2, _mm256_mul_ps(A_vec, B.row[i+1][2]));
        result3 = _mm256_add_ps(result3, _mm256_mul_ps(A_vec, B.row[i+1][3]));

        A_vec = _mm256_broadcast_ss(&A.m[0][i+2]);
        result0 = _mm256_add_ps(result0, _mm256_mul_ps(A_vec, B.row[i+2][0]));
        result1 = _mm256_add_ps(result1, _mm256_mul_ps(A_vec, B.row[i+2][1]));
        result2 = _mm256_add_ps(result2, _mm256_mul_ps(A_vec, B.row[i+2][2]));
        result3 = _mm256_add_ps(result3, _mm256_mul_ps(A_vec, B.row[i+2][3]));

        A_vec = _mm256_broadcast_ss(&A.m[0][i+3]);
        result0 = _mm256_add_ps(result0, _mm256_mul_ps(A_vec, B.row[i+3][0]));
        result1 = _mm256_add_ps(result1, _mm256_mul_ps(A_vec, B.row[i+3][1]));
        result2 = _mm256_add_ps(result2, _mm256_mul_ps(A_vec, B.row[i+3][2]));
        result3 = _mm256_add_ps(result3, _mm256_mul_ps(A_vec, B.row[i+3][3]));
    }

    out.row[0] = result0;
    out.row[1] = result1;
    out.row[2] = result2;
    out.row[3] = result3;
}

void matmult_AVX_1x32x32i(Mat1x32i &out, const Mat1x32i &A, const Mat32x32i &B) {
    _mm256_zeroupper();
    __m256i A_vec = _mm256_set1_epi32(A.m[0][0]);
    __m256i result0 = _mm256_mullo_epi32(A_vec, B.row[0][0]);
    __m256i result1 = _mm256_mullo_epi32(A_vec, B.row[0][1]);
    __m256i result2 = _mm256_mullo_epi32(A_vec, B.row[0][2]);
    __m256i result3 = _mm256_mullo_epi32(A_vec, B.row[0][3]);

    A_vec = _mm256_set1_epi32(A.m[0][1]);
    result0 = _mm256_add_epi32(result0, _mm256_mullo_epi32(A_vec, B.row[1][0]));
    result1 = _mm256_add_epi32(result1, _mm256_mullo_epi32(A_vec, B.row[1][1]));
    result2 = _mm256_add_epi32(result2, _mm256_mullo_epi32(A_vec, B.row[1][2]));
    result3 = _mm256_add_epi32(result3, _mm256_mullo_epi32(A_vec, B.row[1][3]));

    A_vec = _mm256_set1_epi32(A.m[0][2]);
    result0 = _mm256_add_epi32(result0, _mm256_mullo_epi32(A_vec, B.row[2][0]));
    result1 = _mm256_add_epi32(result1, _mm256_mullo_epi32(A_vec, B.row[2][1]));
    result2 = _mm256_add_epi32(result2, _mm256_mullo_epi32(A_vec, B.row[2][2]));
    result3 = _mm256_add_epi32(result3, _mm256_mullo_epi32(A_vec, B.row[2][3]));

    A_vec = _mm256_set1_epi32(A.m[0][3]);
    result0 = _mm256_add_epi32(result0, _mm256_mullo_epi32(A_vec, B.row[3][0]));
    result1 = _mm256_add_epi32(result1, _mm256_mullo_epi32(A_vec, B.row[3][1]));
    result2 = _mm256_add_epi32(result2, _mm256_mullo_epi32(A_vec, B.row[3][2]));
    result3 = _mm256_add_epi32(result3, _mm256_mullo_epi32(A_vec, B.row[3][3]));

    for (int i = 4; i < 32; i+=4) {
        A_vec = _mm256_set1_epi32(A.m[0][i]);
        result0 = _mm256_add_epi32(result0, _mm256_mullo_epi32(A_vec, B.row[i][0]));
        result1 = _mm256_add_epi32(result1, _mm256_mullo_epi32(A_vec, B.row[i][1]));
        result2 = _mm256_add_epi32(result2, _mm256_mullo_epi32(A_vec, B.row[i][2]));
        result3 = _mm256_add_epi32(result3, _mm256_mullo_epi32(A_vec, B.row[i][3]));

        A_vec = _mm256_set1_epi32(A.m[0][i+1]);
        result0 = _mm256_add_epi32(result0, _mm256_mullo_epi32(A_vec, B.row[i+1][0]));
        result1 = _mm256_add_epi32(result1, _mm256_mullo_epi32(A_vec, B.row[i+1][1]));
        result2 = _mm256_add_epi32(result2, _mm256_mullo_epi32(A_vec, B.row[i+1][2]));
        result3 = _mm256_add_epi32(result3, _mm256_mullo_epi32(A_vec, B.row[i+1][3]));

        A_vec = _mm256_set1_epi32(A.m[0][i+2]);
        result0 = _mm256_add_epi32(result0, _mm256_mullo_epi32(A_vec, B.row[i+2][0]));
        result1 = _mm256_add_epi32(result1, _mm256_mullo_epi32(A_vec, B.row[i+2][1]));
        result2 = _mm256_add_epi32(result2, _mm256_mullo_epi32(A_vec, B.row[i+2][2]));
        result3 = _mm256_add_epi32(result3, _mm256_mullo_epi32(A_vec, B.row[i+2][3]));

        A_vec = _mm256_set1_epi32(A.m[0][i+3]);
        result0 = _mm256_add_epi32(result0, _mm256_mullo_epi32(A_vec, B.row[i+3][0]));
        result1 = _mm256_add_epi32(result1, _mm256_mullo_epi32(A_vec, B.row[i+3][1]));
        result2 = _mm256_add_epi32(result2, _mm256_mullo_epi32(A_vec, B.row[i+3][2]));
        result3 = _mm256_add_epi32(result3, _mm256_mullo_epi32(A_vec, B.row[i+3][3]));
    }

    out.row[0] = result0;
    out.row[1] = result1;
    out.row[2] = result2;
    out.row[3] = result3;
}

void matmult_AVX_1x1x32(Mat1x32 &out, const Mat1x32 &A, const Mat1x32 &B) {
    __m256 m_vect = _mm256_broadcast_ss(&A.m[0][0]);
    out.row[0] = _mm256_mul_ps(m_vect, B.row[0]);
    out.row[1] = _mm256_mul_ps(m_vect, B.row[1]);
    out.row[2] = _mm256_mul_ps(m_vect, B.row[2]);
    out.row[3] = _mm256_mul_ps(m_vect, B.row[3]);
}

// template<size_t N>
// inline void matmult_AVX_32x32x32(const Mat32x32 mat1, const Mat32x32 mat2, const Mat32x32 &res) {
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; ++j) {
//             //Stores one element in mat1 and use it in all computations needed before proceeding
//             //Stores as vector to increase computations per cycle
//             vec_mat1 = _mm256_set1_epi32(mat1[i][j]);

//             for (int k = 0; k < N; k += 8) {
//                 vec_mat2 = _mm256_loadu_si256((__m256i*)&mat2[j][k]); //Stores row of second matrix (eight in each iteration)
//                 vec_multi_res = _mm256_loadu_si256((__m256i*)&result[i][k]); //Loads the result matrix row as a vector
//                 vec_multi_res = _mm256_add_epi32(vec_multi_res ,_mm256_mullo_epi32(vec_mat1, vec_mat2));//Multiplies the vectors and adds to th the result vector

//                 _mm256_storeu_si256((__m256i*)&result[i][k], vec_multi_res); //Stores the result vector into the result array
//             }
//         }
//     }
// }


float matmult_AVX_1x32x1(const Mat1x32 &A, const Mat1x32 &B) {

    //__m256i vsum = _mm256_set1_ps(0);
    //float sum = 0;

    __m256 result0 = _mm256_mul_ps(_mm256_broadcast_ss(&A.m[0][0]), B.row[0]);
    result0 = _mm256_add_ps(result0, _mm256_mul_ps(_mm256_broadcast_ss(&A.m[0][0]), B.row[1]));
    result0 = _mm256_add_ps(result0, _mm256_mul_ps(_mm256_broadcast_ss(&A.m[0][0]), B.row[2]));
    result0 = _mm256_add_ps(result0, _mm256_mul_ps(_mm256_broadcast_ss(&A.m[0][0]), B.row[3]));

    // horizontal add of 8 16-bit partial sums and return result
    result0 = _mm256_hadd_ps(result0, result0);
    result0 = _mm256_hadd_ps(result0, result0);
    result0 = _mm256_hadd_ps(result0, result0);
//    result0 = _mm256_hadd_ps(result0, _mm256_srli_si256(result0, 0));
    //sum = result0[0];

    return result0[0];
}

float matmult_ref_1x32x1(const Mat1x32 &A, const Mat1x32 &B) {
    float sum = 0;
    for (int i = 0; i < 32; i++) {
        sum += A.m[0][i] * B.m[0][i];
    }
    return sum;
}


int the_mask = 0; // global so the compiler can't be sure what its value is for opt.

template <typename O, typename I1, typename I2>
static void run_ref(O *out, const I1 *A, const I2 *B, int count) {
    for (int i = 0; i < count; i++) {
        int j = i & the_mask;
        matmult_ref<1, 32>(out[j], A[j], B[j]);
    }
}

static void run_AVX_8(Mat1x32 *out, const Mat1x32 *A, const Mat32x32 *B, int count) {
    for (int i = 0; i < count; i++) {
        int j = i & the_mask;
        matmult_AVX_1x32x32(out[j], A[j], B[j]);
    }
}

static void run_AVX_8i(Mat1x32i *out, const Mat1x32i *A, const Mat32x32i *B, int count) {
    for (int i = 0; i < count; i++) {
        int j = i & the_mask;
        matmult_AVX_1x32x32i(out[j], A[j], B[j]);
    }
}

int main(int argc, char **argv) {

    // Last Layer
    Mat1x32 A0, A1;
    randmat<1, 32, Mat1x32>(A0);
    randmat<1, 32, Mat1x32>(A1);
    float sum = 0;

    for (int i = 0; i < 1; i++) {

        unsigned long long best_time = ~0ull;

        for (int run = 0; run < 1; run++) {
            auto t1 = Clock::now();
            sum = matmult_ref_1x32x1(A0, A1);
            auto t2 = Clock::now();
            auto time = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
            if (time < best_time)
                best_time = time;
        }
        cout << sum << endl;

        double cycles_per_run = (double) best_time;
        printf("%12s: %.2f cycles\n", "Last Layer (Ref)", cycles_per_run);

        for (int run = 0; run < 1; run++) {
            auto t1 = Clock::now();
            float sum = matmult_AVX_1x32x1(A0, A1);
            auto t2 = Clock::now();

            auto time = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
            if (time < best_time)
                best_time = time;
        }
        cout << sum << endl;

        cycles_per_run = (double) best_time;
        printf("%12s: %.2f ns\n", "Last Layer (1x32x1)", cycles_per_run);
    }


    static const struct {
        const char *name;
        void (*matmult)(Mat1x32 &out, const Mat1x32 &A, const Mat32x32 &B);
    } variants[2] = {
            {"matmult_ref_1_32",   matmult_ref<1, 32>},
            {"matmult_AVX_1x32x32", matmult_AVX_1x32x32},
    };

    static const int nvars = (int) (sizeof(variants) / sizeof(*variants));

    srand(1234); // deterministic random tests(TM)

    // correctness tests
    for (int i = 0; i < 1; i++) {
        Mat1x32 A, out, ref_out;
        Mat32x32 B;
        randmat<1, 32, Mat1x32>(A);
        randmat<32, 32, Mat32x32>(B);

        matmult_ref<1, 32>(ref_out, A, B);

        for (int j = 0; j < nvars; j++) {
            // randmat(out);
            variants[j].matmult(out, A, B);
            if (memcmp(&out, &ref_out, sizeof(out)) != 0) {
                fprintf(stderr, "%s fails test\n", variants[j].name);
                exit(1);
            }
        }
    }

    printf("all ok.\n");

    // perf tests
    // as usual with such microbenchmarks, this isn't measuring anything
    // terribly useful, but here goes.
    static const struct {
        const char *name;

        void (*run)(Mat1x32 *out, const Mat1x32 *A, const Mat32x32 *B, int count);
    } perf_variants[2] = {
            {"ref",   run_ref<Mat1x32, Mat1x32, Mat32x32>},
            {"AVX_8", run_AVX_8},
    };
    static const int nperfvars = (int) (sizeof(perf_variants) / sizeof(*perf_variants));

    val = 1;

    Mat1x32 Aperf, out;
    Mat32x32 Bperf;
    randmat<1 ,32>(Aperf);
    randmat<32, 32>(Bperf);

    double t_sum = 0;

    for (int i = 0; i < nperfvars; i++) {
        static const int nruns = 10000;
        static const int muls_per_run = 10000;
        unsigned long long best_time = ~0ull;
        t_sum = 0;
        for (int run = 0; run < nruns; run++) {
            auto t1 = Clock::now();
            perf_variants[i].run(&out, &Aperf, &Bperf, muls_per_run);

            auto t2 = Clock::now();
            auto time = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
            t_sum += (double) time;
            if (time < best_time)
                best_time = time;
        }

        double cycles_per_run = (double) best_time / (double) muls_per_run;
        printf("%15s - min: %.2f ns , avg: %.4f\n", perf_variants[i].name, cycles_per_run, t_sum / (double) muls_per_run / (double) muls_per_run );
    }


    static const struct {
        const char *name;

        void (*run)(Mat1x32i *out, const Mat1x32i *A, const Mat32x32i *B, int count);
    } perf_variantsi[2] = {
            {"ref-integer",   run_ref},
            {"AVX_8-integer", run_AVX_8i},
    };
    static const int nperfvarsi = (int) (sizeof(perf_variantsi) / sizeof(*perf_variantsi));

    Mat1x32i Aperfi, outi;
    Mat32x32i Bperfi;
    randmat<1 ,32>(Aperfi);
    randmat<32, 32>(Bperfi);

    t_sum = 0;

    for (int i = 0; i < nperfvarsi; i++) {
        static const int nruns = 10000;
        static const int muls_per_run = 10000;
        unsigned long long best_time = ~0ull;
        t_sum = 0;
        for (int run = 0; run < nruns; run++) {
            auto t1 = Clock::now();
            perf_variantsi[i].run(&outi, &Aperfi, &Bperfi, muls_per_run);

            auto t2 = Clock::now();
            auto time = chrono::duration_cast<chrono::nanoseconds>(t2 - t1).count();
            t_sum += (double) time;
            if (time < best_time)
                best_time = time;
        }

        double cycles_per_run = (double) best_time / (double) muls_per_run;
        printf("%15s - min: %.2f ns , avg: %.4f\n", perf_variantsi[i].name, cycles_per_run, t_sum / (double) muls_per_run / (double) muls_per_run );
    }

    return 0;
}



