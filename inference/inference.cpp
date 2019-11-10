// g++ -O3 -mavx2 -lpthread -std=c++11 -o inference inference.cpp

#include <fstream>
#include <utility>
#include <cmath>

#include "inference.hpp"
#include "lms_algo.h"
#include "btree.h"

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
int solveSecondLayer(const float &firstLayerOutput, const float &key, 
                const vector<pair<float, float>> &linearModels, const float &N, vector<bool> isModel,
                const unordered_map<int, tree_type> &btreeMap) {
    float temp = firstLayerOutput * linearModels.size() / N;
    int modelIndex = (int) temp; //floor
    if (isModel[modelIndex]) {
        return (key * linearModels[modelIndex].first) + linearModels[modelIndex].second; 
    } else {
        return btree_find(btreeMap.find(modelIndex)->second, key);
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        cout<<"Usage:\ninference <data file> <first layer weights file> <second layer weights>"<<endl;
        exit(0);
    }

    float offset;
    int dataLines;
    vector<int> indices;
    vector<float> data;
    float temp1, temp2;
    int tempInt1, tempInt2;

    ifstream dataFile(argv[1]);
    if (dataFile.is_open()) {
        dataFile>>dataLines;
        for (int i = 0; i < dataLines; ++i) {
            dataFile>>tempInt1;
            dataFile>>temp1;
            if (indices.size() == 0) {
                offset = temp1;
            } else {
                while (tempInt1 != indices.back() + 1) {
                    indices.push_back(indices.back() + 1);
                    data.push_back(data.back());
                }
            }
            indices.push_back(tempInt1);
            data.push_back(temp1-offset);
        }
    }

    Mat1x32 hidden_layer_1;
    Mat32x32 hidden_layer_2;
    Mat1x32 output_layer;
    Mat1x32 key;
    key.m[0][0] = {277647570.899902}; //TODO: randomly test for multiple keys
    float keyToSearch = key.m[0][0];

    ifstream firstLayerWeightsFile(argv[2]);
    if (firstLayerWeightsFile.is_open()) {
        for (int i = 0; i < 32; ++i) {
            firstLayerWeightsFile>>hidden_layer_1.m[0][i];
        }
        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 32; ++j) {
                firstLayerWeightsFile>>hidden_layer_2.m[j][i];
            }
        }
        for (int i = 0; i < 32; ++i) {
            firstLayerWeightsFile>>output_layer.m[0][i];
        }
    }

    ifstream secondLayerWeightsFile(argv[3]);
    float N, modelCount;
    secondLayerWeightsFile>>modelCount>>N;
    vector<pair<float,float>> linearModels;
    vector<pair<float, float>> errors;
    vector<bool> isModel;
    unordered_map<int, tree_type> btreeMap;

    for (int i = 0; i < (int) modelCount; ++i) {
        secondLayerWeightsFile>>temp1>>temp2;
        linearModels.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile>>temp1>>temp2;
        errors.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile>>tempInt1>>tempInt2;
        secondLayerWeightsFile>>temp1;
        if (temp1 == 0.0f) {
            isModel.push_back(false);
            tree_type tree;
            btreeMap[i] = tree;
            btree_insert(tree, data, indices, tempInt1, tempInt2);
        } else {
            isModel.push_back(true);
        }
    }

    float firstLayerAns = solveFirstLayer(hidden_layer_1, hidden_layer_2, output_layer, key);
    cout<<"first layer ans = "<<firstLayerAns<<endl;
    int secondLayerAns = solveSecondLayer(firstLayerAns, keyToSearch, linearModels, data.size(), isModel, btreeMap);
    cout<<"second layer ans = "<<secondLayerAns<<endl;

    int positionOfKey = BinarySearch<float, 512>(data, keyToSearch, midSearchPoint);

    return 0;
}