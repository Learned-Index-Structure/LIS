#include "inference.h"

#include "iaca_mac/iacaMarks.h"

using namespace std;
typedef chrono::high_resolution_clock Clock;

void eval_perf() {
    Mat1x32 key;
    randmat<1, 32, Mat1x32>(key);

    //MM from first hidden layer output
    Mat1x32 out_1;
    Mat1x32 out_2;

    Mat1x32 hidden_layer_1;
    Mat32x32 hidden_layer_2;
    Mat1x32 output_layer;

    randmat<1, 32, Mat1x32>(hidden_layer_1);
    randmat<32, 32, Mat32x32>(hidden_layer_2);
    randmat<1, 32, Mat1x32>(output_layer);

    //Linear regression weight and bias
    float lr_wt[2] = {3.11, 1.32};

    float layer_data[models][32][1] = {{{0}}};
    load_layer_data(layer_data);

    float position = 0.0;
    long long sum = 0;
    auto t1 = Clock::now();

    for (int i = 0; i < PREDICT_ITER; ++i) {
        matmult_AVX_1x1x32(out_1, key, hidden_layer_1);
        relu<1, 32>(out_1);
        matmult_AVX_1x32x32(out_2, out_1, hidden_layer_2);
        relu<1, 32>(out_2);
        position = matmult_AVX_1x32x1(out_2, output_layer);

        float pred = position * models / N;
        float value = pred * lr_wt[0] + lr_wt[1];
        sum += (long) value;
    }

    auto t2 = Clock::now();
    cout << "Time: "
         << chrono::duration<int64_t, std::nano>(t2 - t1).count() / PREDICT_ITER
         << " nanoseconds" << std::endl;

    cout << "Key, Value: " << key.m[0][0] << ", " << sum << endl;
}


int main(int argc, char **argv) {
    eval_perf();
    return 0;
}