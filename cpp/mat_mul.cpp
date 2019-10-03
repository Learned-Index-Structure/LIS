#include<time.h>
#include<gsl/gsl_blas.h>
#include<omp.h>
#include<iostream>

using namespace std;

//https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm
//https://www.physicsforums.com/threads/understanding-blas-dgemm-in-c.543110/

//  gcc mat_mul.cpp -L/usr/local/lib -lgsl -lgslcblas -o mat_mul
//  g++ mat_mul.cpp -L/usr/local/lib -lgsl -lgslcblas -o mat_mul
//  g++-9  mat_mul.cpp -I '/usr/local/include' -L/usr/local/lib -lgsl -lgslcblas -o mat_mul
//  g++-9  mat_mul.cpp -I '/usr/local/include' -L/usr/local/lib -lgsl -lgslcblas -o mat_mul -fopenmp

void print_matrix(const double *A, const int m, const int n) {

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[j + i * n] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

int main() {
    enum CBLAS_ORDER order;
    enum CBLAS_TRANSPOSE trasnsa;
    enum CBLAS_TRANSPOSE trasnsb;

    order = CblasRowMajor; // Defines the layout in which elements are stored in 2D array.
    trasnsa = CblasNoTrans;
    trasnsb = CblasNoTrans;

    double *A, *B, *C;

    double alpha, beta;

    int m, n, k, lda, ldb, ldc, i;

    m = 3;
    n = 4;
    k = 5;

    alpha = 1;
    beta = 0;

    //Strides of the matrix
    lda = 4;
    ldb = 5;
    ldc = 5;

    A = new double[m * n];
    B = new double[n * k];
    C = new double[m * k];

    // Initialize the matrix
    for (i = 0; i < m * n; i++) {
        A[i] = 1;
    }
    for (i = 0; i < n * k; i++) {
        B[i] = i + 1;
    }
    for (i = 0; i < m * k; i++) {
        C[i] = 0.0;
    }


    struct timespec tstart = {0, 0}, tend = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    cblas_dgemm(order, trasnsa, trasnsb, m, k, n, alpha, A, lda, B, ldb, beta, C, ldc);

    clock_gettime(CLOCK_MONOTONIC, &tend);

    print_matrix(A, m, n);
    print_matrix(B, n, k);
    print_matrix(C, m, k);

    printf("\nTime taken : %ld ns \n", (tend.tv_nsec - tstart.tv_nsec));

    return 0;
}


