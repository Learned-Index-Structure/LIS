// #include "/usr/local/include/gsl/gsl_blas.h"
#include<time.h>
#include<gsl/gsl_blas.h>
#include<omp.h>
#include<iostream>


using namespace std;
//  gcc mat_mul.cpp -L/usr/local/lib -lgsl -lgslcblas -o mat_mul
//  g++ mat_mul.cpp -L/usr/local/lib -lgsl -lgslcblas -o mat_mul
//  g++-9  mat_mul.cpp -I '/usr/local/include' -L/usr/local/lib -lgsl -lgslcblas -o mat_mul
//  g++-9  mat_mul.cpp -I '/usr/local/include' -L/usr/local/lib -lgsl -lgslcblas -o mat_mul -fopenmp

// extern void sgemm_( char *, char *, int *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int * );

//extern int gsl_blas_sgemm(CBLAS_TRANSPOSE_t TransA,
//                          CBLAS_TRANSPOSE_t TransB,
//                          float alpha,
//                          const gsl_matrix_float *A,
//                          const gsl_matrix_float *B,
//                          float beta,
//                          gsl_matrix_float *C);

int main() {
    enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE trasnsa;
    enum CBLAS_TRANSPOSE trasnsb;

    order = CblasColMajor;
    trasnsa = CblasNoTrans;
    trasnsb = CblasNoTrans;

    double *A, *B, *C;

    double alpha, beta;

    int m, n, k, lda, ldb, ldc, i;

    int size = 64;
    m = size;
    n = size;
    k = size;
    alpha = 1;
    beta = 0;

    lda = size;
    ldb = size;
    ldc = size;

    // A = (double *)malloc(sizeof(double) * m * n);
    // B = (double *)malloc(sizeof(double) * m * n);
    // C = (double *)malloc(sizeof(double) * m * n);

    A = new double[m * n];
    B = new double[m * n];
    C = new double[m * n];

    #pragma omp parallel num_threads(8)
    {
        // int id = omp_get_thread_num();
        // int total = omp_get_num_threads();
        // std::cout << "id: " << id << "total: " << total << "\n" <<endl;

        for (i = 0; i < m * n; i++) {
            A[i] = 1.0;
            B[i] = 2.0;
            C[i] = 0.0;
        }
    }

    struct timespec tstart = {0, 0}, tend = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    cblas_dgemm(order, trasnsa, trasnsb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    clock_gettime(CLOCK_MONOTONIC, &tend);

    printf("\n\nTime taken : %ld ns \n", (tend.tv_nsec - tstart.tv_nsec));
//    printf("\n\nTime taken is %.9f s \n\n", ((double) tend.tv_sec + 1.0e-9 * tend.tv_nsec) -
//                                            ((double) tstart.tv_sec + 1.0e-9 * tstart.tv_nsec));

//    for (i = 0; i < m * n; i++) {
//        printf(" %d = %f ", i, C[i]);
//    }

//    gsl_matrix_float a;
//    gsl_matrix_float b;
//    gsl_matrix_float c;

//    gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1.0, &a, &b, 0, &c);
    return 0;
}
