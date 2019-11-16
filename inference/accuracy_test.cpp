#include "inference.hpp"

using namespace std;
typedef chrono::high_resolution_clock Clock;

inline double matmult_AVX_1x16x1_Naive(const Mat1x16 &A, const Mat1x16 &B) {
    double result = 0.0f;
    for (int i = 0; i < 16; ++i) {
        result += A.m[0][i] * B.m[0][i];
    }

    return result;
}

inline double matmult_AVX_1x32x1_Naive(const Mat1x32d &A, const Mat1x32d &B) {

    double result = 0.0f;
    for (int i = 0; i < 32; ++i) {
        result += A.m[0][i] * B.m[0][i];
    }

    return result;
}

static void escape(void *p) {
    asm volatile("" : : "g"(p) : "memory");
}

static void clobber() {
    asm volatile("" : : : "memory");
}


int main() {
    int N_ITER = 1000000;

    Mat1x32d A, B, out_avx, out_naive;
    Mat32x32d C;

    for (int i = 0; i < N_ITER; i++) {
        randmatd<1, 32>(A);
        randmatd<1, 32>(B);

        matmult_AVX_1x1x32d(out_avx, A, B);
        MatMulNaive<1, 1, 32>(out_naive, A, B);

        if (memcmp(&out_avx, &out_naive, sizeof(out_avx)) != 0) {
            std::cout << "iter : " << i << std::endl;
            exit(1);
        }
    }
    std::cout << "First Layer Passed." << std::endl;

    for (int i = 0; i < N_ITER; i++) {
        randmatd<1, 32>(A);
        randmatd<32, 32>(C);

        matmult_AVX_1x32x32d(out_avx, A, C);
        MatMulNaive<1, 32, 32>(out_naive, A, C);

        if (memcmp(&out_avx, &out_naive, sizeof(out_avx)) != 0) {
            std::cout << "iter : " << i << std::endl;
            exit(1);
        }
    }
    std::cout << "Second Layer Passed." << std::endl;

    for (int i = 0; i < N_ITER; i++) {
        randmatd<1, 32>(A);
        randmatd<1, 32>(B);

        double a = matmult_AVX_1x32x1_REFd(A, B);
        double b = matmult_AVX_1x32x1_Naive(A, B);

        if (a - b > 1e-5) {
            std::cout << "iter : " << i << std::endl;
            std::cout << "a, b " << a << " " << b << " " << (a - b) << std::endl;
            exit(1);
        }
    }
    std::cout << "Third Layer Passed." << std::endl;


    //Matrix Multiplication Runtime
    long sum = 0;
    auto t1 = Clock::now();
    for(int i =0; i < N_ITER; i++) {
        randmatd<1, 32>(A);
        randmatd<1, 32>(B);
        randmatd<32, 32>(C);

        matmult_AVX_1x1x32d(out_avx, A, B);
        relu<1, 32>(out_avx);
        matmult_AVX_1x32x32d(out_avx, A, C);
        relu<1, 32>(out_avx);
        sum  += matmult_AVX_1x32x1_REFd(A, B);
    }

    auto t2 = Clock::now();
    std::cout << "Time: "
              << (chrono::duration<int64_t, std::nano>(t2 - t1).count() / N_ITER)
              << " nanoseconds" << std::endl;
    cout << sum << endl;


    //Matrix Initialization Runtime
    sum = 0;
    t1 = Clock::now();
    for(int i =0; i < N_ITER; i++) {
        escape(&A);
        escape(&B);
        escape(&C);
        randmatd<1, 32>(A);
        randmatd<1, 32>(B);
        randmatd<32, 32>(C);
        clobber();
    }

    t2 = Clock::now();
    std::cout << "LOAD Time: "
              << (chrono::duration<int64_t, std::nano>(t2 - t1).count() / N_ITER)
              << " nanoseconds" << std::endl;
    cout << sum << endl;



    // 16-Neurons Accuracy test
    {
        cout << "16x16" << endl;
        Mat1x16 A, B, out_avx, out_naive;
        Mat16x16 C;

        for (int i = 0; i < N_ITER; i++) {
            randmatd<1, 16>(A);
            randmatd<1, 16>(B);

            matmult_AVX_1x1x16(out_avx, A, B);
            MatMulNaive<1, 1, 16>(out_naive, A, B);

            if (memcmp(&out_avx, &out_naive, sizeof(out_avx)) != 0) {
                std::cout << "iter : " << i << std::endl;
                exit(1);
            }
        }
        std::cout << "First Layer Passed." << std::endl;

        for (int i = 0; i < N_ITER; i++) {
            randmatd<1, 16>(A);
            randmatd<16, 16>(C);

            matmult_AVX_1x16x16(out_avx, A, C);
            MatMulNaive<1, 16, 16>(out_naive, A, C);

            if (memcmp(&out_avx, &out_naive, sizeof(out_avx)) != 0) {
                std::cout << "iter : " << i << std::endl;
                exit(1);
            }
        }
        std::cout << "Second Layer Passed." << std::endl;

        for (int i = 0; i < N_ITER; i++) {
            randmatd<1, 32>(A);
            randmatd<1, 32>(B);

            double a = matmult_AVX_1x16x1_REF(A, B);
            double b = matmult_AVX_1x16x1_Naive(A, B);

            if (a - b > 1e-5) {
                std::cout << "iter : " << i << std::endl;
                std::cout << "a, b " << a << " " << b << " " << (a - b) << std::endl;
                exit(1);
            }
        }
        std::cout << "Third Layer Passed." << std::endl;


        //Matrix Multiplication Runtime
        long sum = 0;
        auto t1 = Clock::now();
        for(int i =0; i < N_ITER; i++) {
            randmatd<1, 16>(A);
            randmatd<1, 16>(B);
            randmatd<16, 16>(C);

            matmult_AVX_1x1x16(out_avx, A, B);
            relu<1, 16>(out_avx);
            matmult_AVX_1x16x16(out_avx, A, C);
            relu<1, 16>(out_avx);
            sum  += matmult_AVX_1x16x1_REF(A, B);
        }

        auto t2 = Clock::now();
        std::cout << "Time: "
                  << (chrono::duration<int64_t, std::nano>(t2 - t1).count() / N_ITER)
                  << " nanoseconds" << std::endl;
        cout << sum << endl;


        //Matrix Initialization Runtime
        sum = 0;
        t1 = Clock::now();
        for(int i =0; i < N_ITER; i++) {
            escape(&A);
            escape(&B);
            escape(&C);
            randmatd<1, 16>(A);
            randmatd<1, 16>(B);
            randmatd<16, 16>(C);
            clobber();
        }

        t2 = Clock::now();
        std::cout << "LOAD Time: "
                  << (chrono::duration<int64_t, std::nano>(t2 - t1).count() / N_ITER)
                  << " nanoseconds" << std::endl;
        cout << sum << endl;

    }
}
