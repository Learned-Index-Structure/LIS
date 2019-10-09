#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace std;
typedef chrono::high_resolution_clock Clock;

template<size_t rows, size_t mid, size_t cols>
void mm_normal(double (&A)[rows][mid], double (&B)[mid][cols], double (&C)[rows][cols]) {
    for (int z = 0; z < rows; ++z) {
        for (int j = 0; j < cols; ++j) {
            for (int o = 0; o < mid; ++o) {
                C[z][j] += A[z][o] * B[o][j];
            }
        }
    }
}

template<size_t rows, size_t cols>
void get_weight(double (&array)[rows][cols]) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            array[i][j] = rand() % 10 + 1.0;
        }
    }
}

template<size_t rows, size_t cols>
void print_matrix(double (&array)[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << array[i][j] << '\t';
        }
        std::cout << std::endl;
    }
}

template<size_t models, size_t rows, size_t cols>
void load_layer_data(double (&layer_data)[models][rows][cols]) {
    for (int i = 0; i < models; i++) {
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                layer_data[i][j][k] = rand() % 10 + 1.0;
            }
        }
    }
}

template<size_t rows, size_t cols>
void relu(double (&data)[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = (((unsigned int) data[i][j] & 0x80000000) != 0) ? data[i][j] : 0;
        }
    }
}

#define models 10000
#define N 1000000000

void eval() {
    double key[1][1] = {{4.0}};

    //MM from first hidden layer output
    double out_1[1][32] = {{0.0}};
    double out_2[1][32] = {{0.0}};

    double hidden_layer_1[1][32] = {{0.0}};
    double hidden_layer_2[32][32] = {{0.0}};
    double output_layer[32][1] = {{0.0}};

    get_weight(hidden_layer_1);
    get_weight(hidden_layer_2);
    get_weight(output_layer);

    //Linear regression weight and bias
    double lr_wt[2] = {3.42, 1.44};

    double layer_data[models][32][1] = {{{0}}};
    load_layer_data(layer_data);

    double pred_layer_1[1][1] = {{0.0}};

//    unsigned long long tstart = __rdtsc();

    auto t1 = Clock::now();
    mm_normal(key, hidden_layer_1, out_1);
    relu(out_1);
    mm_normal(out_1, hidden_layer_2, out_2);
    relu(out_2);
    mm_normal(out_2, output_layer, pred_layer_1);
    relu(pred_layer_1);

    int pred = pred_layer_1[0][0] * models / N;

    double value = pred * lr_wt[0] + lr_wt[1];
    auto t2 = Clock::now();
//    unsigned long long tend = __rdtsc();
    std::cout << "Time: "
              << chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()
              << " nanoseconds" << std::endl;

    std::cout << "Key, Value: " << key[0][0] << ", " << value << std::endl;
//    std::cout << "Time: " << (tend - tstart) * 0.36873156342 << std::endl; //0.36873156342 = 1 / clock_speed
}

int main(int argc, char *argv[]) {
    eval();

    if (argc != 1) {
        cout << "Usage:" << endl;
        cout << "./infer [first layer weights file]" << endl;
        exit(0);
    }

//    string firstLayerWeights = argv[1];
    return 0;
}
