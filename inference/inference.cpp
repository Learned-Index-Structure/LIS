#include <fstream>
#include <utility>

#include "inference.h"

#include "iaca_mac/iacaMarks.h"

using namespace std;
typedef chrono::high_resolution_clock Clock;

inline
float solveFirstLayer(const Mat1x32 &hidden_layer_1, const Mat32x32 &hidden_layer_2, const Mat1x32 &output_layer, const Mat1x32 &key) {
    //MM from first hidden layer output
    Mat1x32 out_1;
    Mat1x32 out_2;

    matmult_AVX_1x1x32(out_1, key, hidden_layer_1);
    relu<1, 32>(out_1);
    matmult_AVX_1x32x32(out_2, out_1, hidden_layer_2);
    relu<1, 32>(out_2);
    return matmult_AVX_1x32x1(out_2, output_layer);
}

inline
float solveSecondLayer(const float &firstLayerOutput, const vector<pair<int, int>> &linearModels, const int &N) {
    int modelIndex = firstLayerOutput * linearModels.size() / N;
    return (firstLayerOutput * linearModels[i].first) + linearModels[i].second;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        cout<<"Usage:\ninference <first layer weights file> <second layer weights>"<endl;
        exit(0);
    }

    Mat1x32 hidden_layer_1;
    Mat32x32 hidden_layer_2;
    Mat1x32 output_layer;
    Mat1x32 key;

    ifstream firstLayerWeightsFile(argv[1]);
    if (firstLayerWeightsFile.is_open()) {
        for (int i = 0; i < 32; ++i) {
            firstLayerWeightsFile>>hidden_layer_1.m[0][i];
        }
        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 32; ++j) {
                firstLayerWeightsFile>>hidden_layer_2.m[i][j];
            }
        }
        for (int i = 0; i < 32; ++i) {
            firstLayerWeightsFile>>output_layer.m[0][i];
        }
    }

    ifstream secondLayerWeightsFile(argv[2]);
    int N, modelCount;
    ifstream>>N>>modelCount;
    vector<pair<int,int>> linearModels;
    vector<pair<int, int>> errors;
    vector<bool> isModel;
    float temp1, temp2;

    for (int i = 0; i < modelCount; ++i) {
        secondLayerWeightsFile>>temp1>>temp2;
        linearModels.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile>>temp1>>temp2;
        errors.push_back(temp1, temp2);
        secondLayerWeightsFile>>temp1>>temp2;
        secondLayerWeightsFile>>temp1;
        if (temp1 == 0.0f) {
            isModel.push_back(false);
        } else {
            isModel.push_back(true);
        }
    }

    float firstLayerAns = solveFirstLayer(hidden_layer_1, hidden_layer_2, output_layer, key);
    float secondLayerAns = solveSecondLayer(firstLayerAns, linearModels, N);

    return 0;
}