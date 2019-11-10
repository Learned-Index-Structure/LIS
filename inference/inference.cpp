// g++ -O3 -mavx2 -lpthread -std=c++11 -o inference inference.cpp

#include <fstream>
#include <utility>
#include <cmath>
#include <unordered_map>

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
    return matmult_AVX_1x32x1_REF(out_2, output_layer);
}

inline
int solveSecondLayer(const float &firstLayerOutput, const float &key, 
                const vector<pair<float, float>> &linearModels, const float &N, vector<bool> isModel,
                const unordered_map<int, tree_type> &btreeMap) {
    float temp = firstLayerOutput * linearModels.size() / N;
    int modelIndex = (int) temp; //floor
    cout<<"modelIndex = "<<modelIndex<<endl;
    //TODO: remove this branch
    if (isModel[modelIndex]) {
        cout<<"linear regression"<<endl;
        cout<<" m = "<<linearModels[modelIndex].first<<" c = "<<linearModels[modelIndex].second<<endl;
        float temp2 = (key * linearModels[modelIndex].first) + linearModels[modelIndex].second; 
        cout<<"float lr ans = "<<temp2<<endl;
        return temp2;
    } else {
        cout<<"btree"<<endl;
        return btree_find(btreeMap.find(modelIndex)->second, key);
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        cout<<"Usage:\ninference <data file> <first layer weights file> <second layer weights>"<<endl;
        exit(0);
    }

    double offset;
    int dataLines;
    vector<int> indices;
    vector<float> data;
    float temp1, temp2, temp3;
    int tempInt1, tempInt2;
    double tempDouble;

    ifstream dataFile(argv[1]);
    if (dataFile.is_open()) {
        dataFile>>dataLines;
        for (int i = 0; i < dataLines; ++i) {
            dataFile>>tempInt1;
            dataFile>>tempDouble;
            if (indices.size() == 0) {
                offset = tempDouble;
            } else {
                while (tempInt1 != indices.back() + 1) {
                    indices.push_back(indices.back() + 1);
                    data.push_back(data.back());
                }
            }
            indices.push_back(tempInt1);
            data.push_back(tempDouble-offset);
            void *bits = &tempDouble;
            if (i < 100) {
                cout<<"orig = "<<tempDouble<<" actual = "<<data.back()<<" bits = "<<*((int*)bits)<<endl;
            }
        }
    }
    dataFile.close();

    Mat1x32 hidden_layer_1;
    Mat32x32 hidden_layer_2;
    Mat1x32 output_layer;
    Mat1x32 key;
    key.m[0][0] = {5306.900391}; //TODO: randomly test for multiple keys
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
    firstLayerWeightsFile.close();

    ifstream secondLayerWeightsFile(argv[3]);
    float N, modelCount;
    int threshold;
    secondLayerWeightsFile>>modelCount>>N>>temp1>>temp2;
    threshold = (int) temp2;
    vector<pair<float,float>> linearModels;
    vector<pair<float, float>> errors;
    vector<bool> isModel;
    unordered_map<int, tree_type> btreeMap;

    for (int i = 0; i < (int) modelCount; ++i) {
        secondLayerWeightsFile>>temp1>>temp2;
        linearModels.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile>>temp1>>temp2;
        errors.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile>>temp1>>temp2;
        secondLayerWeightsFile>>temp3;
        if (temp3 == 0.0f) {
            isModel.push_back(false);
            tree_type tree;
            btreeMap[i] = tree;
            btree_insert(tree, data, indices, (int)temp1, (int)temp2);
        } else {
            isModel.push_back(true);
        }
    }
    secondLayerWeightsFile.close();

    float firstLayerAns = solveFirstLayer(hidden_layer_1, hidden_layer_2, output_layer, key);
    cout<<"first layer ans = "<<firstLayerAns<<endl;
    int secondLayerAns = solveSecondLayer(firstLayerAns, keyToSearch, linearModels, data.size(), isModel, btreeMap);
    cout<<"second layer ans = "<<secondLayerAns<<endl;

    cout<<"threshold = "<<threshold<<endl;
    int positionOfKey = binarySearchBranchless<float>(data, keyToSearch, secondLayerAns, threshold);
    cout<<"position of key = "<<positionOfKey<<" value = "<<data[positionOfKey]<<endl;

    return 0;
}