#include "inference.hpp"

inline double matmult_AVX_1x32x1_Naive(const Mat1x32d &A, const Mat1x32d &B) {

    double result = 0.0f;
    for (int i = 0; i < 32; ++i) {
        result += A.m[0][i] * B.m[0][i];
    }

    return result;
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


}
