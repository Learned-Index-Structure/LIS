#include <iostream>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <string>

using namespace std;

typedef chrono::high_resolution_clock Clock;

#define PREDICT_ITER 1000ll

template<size_t rows, size_t mid, size_t cols>
void mm_normal(int (&A)[rows][mid], int (&B)[mid][cols], int (&C)[rows][cols]) {
    for (int z = 0; z < rows; ++z) {
        for (int j = 0; j < cols; ++j) {
            for (int o = 0; o < mid; ++o) {
                C[z][j] += A[z][o] * B[o][j];
            }
        }
    }
}

template<size_t rows, size_t cols>
void get_weight(int (&array)[rows][cols]) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            array[i][j] = rand() % 10 + 1.0;
        }
    }
}

template<size_t rows, size_t cols>
void print_matrix(int (&array)[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            std::cout << array[i][j] << '\t';
        }
        std::cout << std::endl;
    }
}

template<size_t models, size_t rows, size_t cols>
void load_layer_data(int (&layer_data)[models][rows][cols]) {
    for (int i = 0; i < models; i++) {
        for (int j = 0; j < rows; j++) {
            for (int k = 0; k < cols; k++) {
                layer_data[i][j][k] = rand() % 10 + 1.0;
            }
        }
    }
}

template<size_t rows, size_t cols>
void relu(int (&data)[rows][cols]) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            data[i][j] = (((unsigned int) data[i][j] & 0x80000000) != 0) ? data[i][j] : 0;
        }
    }
}

#define models 10000
#define N 1000000000

void eval() {
    int key[1][1] = {{4}};

    //MM from first hidden layer output
    int out_1[1][32] = {{0}};
    int out_2[1][32] = {{0}};

    int hidden_layer_1[1][32] = {{0}};
    int hidden_layer_2[32][32] = {{0}};
    int output_layer[32][1] = {{0}};

    get_weight(hidden_layer_1);
    get_weight(hidden_layer_2);
    get_weight(output_layer);

    //Linear regression weight and bias
    int lr_wt[2] = {3, 1};

    int layer_data[models][32][1] = {{{0}}};
    load_layer_data(layer_data);

    int pred_layer_1[1][1] = {{0}};

    auto t1 = Clock::now();
    for (int i = 0; i < PREDICT_ITER; ++i) {
        mm_normal(key, hidden_layer_1, out_1);
        relu(out_1);
        mm_normal(out_1, hidden_layer_2, out_2);
        relu(out_2);
        mm_normal(out_2, output_layer, pred_layer_1);
        relu(pred_layer_1);
    }

    int pred = pred_layer_1[0][0] * models / N;

    double value = pred * lr_wt[0] + lr_wt[1];
    auto t2 = Clock::now();
    std::cout << "Time: "
              << chrono::duration<int64_t, std::nano>(t2 - t1).count()/PREDICT_ITER
              << " nanoseconds" << std::endl;

    std::cout << "Key, Value: " << key[0][0] << ", " << value << std::endl;
//    std::cout << "Time: " << (tend - tstart) * 0.36873156342 << std::endl; //0.36873156342 = 1 / clock_speed
}

int main(int argc, char *argv[]) {
    eval();

    exit(0);

    if (argc != 1) {
        cout << "Usage:" << endl;
        cout << "./infer [first layer weights file]" << endl;
        exit(0);
    }

    string firstLayerWeightsFile = argv[1];
    ifstream firstLayerWeightsStream(firstLayerWeightsFile);
    //n: number of neurons in one layer of top NN
    //m: number of layers in the top NN
    int n, m;
    firstLayerWeightsStream>>n>>m;

    int w1[n], w2[n][n], w3[n];
    for (int i = 0; i < n; ++i) {
        firstLayerWeightsStream>>w1[i];
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            firstLayerWeightsStream>>w2[i][j];
        }
    }
    for (int i = 0; i < n; ++i) {
        firstLayerWeightsStream>>w3[i];
    }

    return 0;
}
