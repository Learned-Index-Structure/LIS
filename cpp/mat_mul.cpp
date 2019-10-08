#include<time.h>
#include<gsl/gsl_blas.h>
#include<omp.h>
#include<iostream>
#include<chrono>

// #include <x86intrin.h>
// #include<intrin.h>
// #include <immintrin.h>

using namespace std;
using namespace chrono;

//https://software.intel.com/en-us/mkl-tutorial-c-multiplying-matrices-using-dgemm
//https://www.physicsforums.com/threads/understanding-blas-dgemm-in-c.543110/

//  gcc mat_mul.cpp -L/usr/local/lib -lgsl -lgslcblas -o mat_mul
//  g++ mat_mul.cpp -L/usr/local/lib -lgsl -lgslcblas -o mat_mul
//  g++-9  mat_mul.cpp -I '/usr/local/include' -L/usr/local/lib -lgsl -lgslcblas -o mat_mul
//  g++-9  mat_mul.cpp -I '/usr/local/include' -L/usr/local/lib -lgsl -lgslcblas -o mat_mul -fopenmp



#if defined(__i386__)

static __inline__ unsigned long long __rdtsc(void)
{
    unsigned long long int x;
    __asm__ volatile (".byte 0x0f, 0x31" : "=A" (x));
    return x;
}

#elif defined(__x86_64__)

__inline__ unsigned long long __rdtsc(void)
{
    unsigned hi, lo;
    __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
    return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

#endif

__inline__ uint64_t rdtsc(void) {
  uint32_t lo, hi;
  __asm__ __volatile__ (      // serialize
  "xorl %%eax,%%eax \n        cpuid"
  ::: "%rax", "%rbx", "%rcx", "%rdx");
  /* We cannot use "=A", since this would use %rax on x86_64 and return only the lower 32bits of the TSC */
  __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
  return (uint64_t)hi << 32 | lo;
}

void mm_naive(int num, int m , int n, int k) {

    int i =0;
     struct timespec tstart = {0, 0}, tend = {0, 0};

    double AA[m][n], BB[n][k], CC[m][k];

    for (i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            AA[i][j] = num + i;
        }
    }
    int cnt = 1;
    for (i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            BB[i][j] = num + 1;
        }
    }

    for (i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            CC[i][j] = 0.0;
        }
    }

    // tstart = {0, 0}, tend = {0, 0};
    // clock_gettime(CLOCK_MONOTONIC, &tstart);
    unsigned long long time = __rdtsc();
    // high_resolution_clock::time_point t1 = high_resolution_clock::now();
    // omp_set_num_threads(m*n*k);
    // #pragma omp parallel num_threads(m)
    // {

    // __m256 increment = _mm256_load_ps(AA);
    // #pragma omp for simd
    for (int z = 0; z < m; ++z) {
        // #pragma omp for simd
        for (int j = 0; j < k; ++j) {
            // #pragma omp for simd
            for (int o = 0; o < n; ++o) {
                CC[z][j] += AA[z][o] * BB[o][j];
            }
        }
    }

    // high_resolution_clock::time_point t2 = high_resolution_clock::now();
    // cout <<"\nCHRONO Time: " <<  duration_cast<nanoseconds>(t2 - t1).count();
    // cout  << "\nCHRONO Time: " << duration_cast<nanoseconds>(t2 - t1).count();
    unsigned long long endtime = __rdtsc();
    // clock_gettime(CLOCK_MONOTONIC, &tend);
    cout << "\nRDTSC: time : " << (endtime -  time) * 0.37 << " ns";
    // printf("\nCLOCK: Time taken : %ld ns \n", (tend.tv_nsec - tstart.tv_nsec));

    // }


}
void print_matrix(const double *A, const int m, const int n) {

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            cout << A[j + i * n] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

//void init(double AA[][], int m, int n) {
//    for (int i = 0; i < m; i++) {
//        for (int i = 0; i < m; i++) {
//            AA[i][j] = 1.0;
//        }
//    }
//}

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

    m = 1;
    n = 32;
    k = 1;

    alpha = 1;
    beta = 0;

    //Strides of the matrix
    lda = n;
    ldb = k;
    ldc = k;

    A = new double[m * n];
    B = new double[n * k];
    C = new double[m * k];

    // Initialize the matrix
    for (i = 0; i < m * n; i++) {
        A[i] = i;
    }
    for (i = 0; i < n * k; i++) {
        B[i] = i+1;
    }
    for (i = 0; i < m * k; i++) {
        C[i] = 0.0;
    }

    struct timespec tstart = {0, 0}, tend = {0, 0};
    clock_gettime(CLOCK_MONOTONIC, &tstart);

    cblas_dgemm(order, trasnsa, trasnsb, m, k, n, alpha, A, lda, B, ldb, beta, C, ldc);

    clock_gettime(CLOCK_MONOTONIC, &tend);

     printf("\nBLAS Time taken : %ld ns \n", (tend.tv_nsec - tstart.tv_nsec));
//    print_matrix(A, m, n);
//    print_matrix(B, n, k);
//    print_matrix(C, m, k);
    mm_naive(1, m, n, k);
    mm_naive(2, m, n, k);
    mm_naive(3, m, n, k);
    mm_naive(4, m, n, k);
    mm_naive(1, m, n, k);
    mm_naive(2, m, n, k);
    mm_naive(3, m, n, k);
    mm_naive(4, m, n, k);
    mm_naive(1, m, n, k);
    mm_naive(2, m, n, k);
    mm_naive(3, m, n, k);
    mm_naive(4, m, n, k);
    mm_naive(1, m, n, k);
    mm_naive(2, m, n, k);
    mm_naive(3, m, n, k);
    mm_naive(4, m, n, k);
    mm_naive(1, m, n, k);
    mm_naive(2, m, n, k);
    mm_naive(3, m, n, k);
    mm_naive(4, m, n, k);
    return 0;
}


