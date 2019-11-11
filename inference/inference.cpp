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
float solveFirstLayer(const Mat1x32 &hidden_layer_1, const Mat32x32 &hidden_layer_2, const Mat1x32 &output_layer,
                      const Mat1x32 &key) {
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
pair<int, float> solveSecondLayer(const float &firstLayerOutput, const float &key,
                                  const vector<pair<float, float> > &linearModels, const float &N, vector<bool> isModel,
                                  const unordered_map<int, tree_type> &btreeMap) {
    float temp = firstLayerOutput * linearModels.size() / N;
    int modelIndex = (int) temp; //floor
    cout << "modelIndex = " << modelIndex << endl;

    cout << "linear regression" << endl;
    cout << " m = " << linearModels[modelIndex].first << " c = " << linearModels[modelIndex].second << endl;
    float temp2 = (key * linearModels[modelIndex].first) + linearModels[modelIndex].second;
    cout << "float lr ans = " << temp2 << endl;
    return make_pair(modelIndex, temp2);

}

int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "Usage:\ninference <data file> <first layer weights file> <second layer weights>" << endl;
        exit(0);
    }

    double offset;
    int dataLines;
    vector<uint32_t> indices;
    vector<double> data;
    float temp1, temp2, temp3;
    uint32_t tempInt1, tempInt2;
    double tempDouble;

    vector<double, uint32_t> keyList;

    ifstream dataFile(argv[1]);
    if (dataFile.is_open()) {
        dataFile >> dataLines;
        for (int i = 0; i < dataLines; ++i) {
            dataFile >> tempInt1;
            dataFile >> tempDouble;
            
            if (i % 10 == 0)
                keyList.push_back(tempDouble, tempInt1);

            if (indices.size() == 0) {
                offset = tempDouble;
            } else {
                while (i < dataLines && tempInt1 != indices.back() + 1) {
                    indices.push_back(indices.back() + 1);
                    data.push_back(data.back());
                }
            }
            indices.push_back(tempInt1);
            data.push_back(tempDouble - offset);
        }
    }
    dataFile.close();
    cout << "offset: " << offset << endl;

    Mat1x32 hidden_layer_1;
    Mat32x32 hidden_layer_2;
    Mat1x32 output_layer;
    Mat1x32 key;
//    key.m[0][0] = 1427781637776.1 - offset; //TODO: randomly test for multiple keys
    key.m[0][0] = 1427787916750.0 - offset;

//    float keyToSearch = key.m[0][0];
    double keyToSearch = 1427787916750.0 - offset;

    ifstream firstLayerWeightsFile(argv[2]);
    if (firstLayerWeightsFile.is_open()) {
        for (int i = 0; i < 32; ++i) {
            firstLayerWeightsFile >> hidden_layer_1.m[0][i];
        }
        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 32; ++j) {
                firstLayerWeightsFile >> hidden_layer_2.m[j][i];
            }
        }
        for (int i = 0; i < 32; ++i) {
            firstLayerWeightsFile >> output_layer.m[0][i];
        }
    }
    firstLayerWeightsFile.close();

    ifstream secondLayerWeightsFile(argv[3]);
    float N, modelCount;
    int threshold;
    secondLayerWeightsFile >> modelCount >> N >> temp1 >> temp2;
    threshold = (int) temp2;
    vector<pair<float, float> > linearModels;
    vector<pair<float, float> > errors;
    vector<bool> isModel;
    unordered_map<int, tree_type> btreeMap;

    for (int i = 0; i < (int) modelCount; ++i) {
        secondLayerWeightsFile >> temp1 >> temp2;
        linearModels.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile >> temp1 >> temp2;
        errors.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile >> temp1 >> temp2;
        secondLayerWeightsFile >> temp3;

        isModel.push_back(temp3 != 0.0f);
    }
    secondLayerWeightsFile.close();

    float firstLayerAns = solveFirstLayer(hidden_layer_1, hidden_layer_2, output_layer, key);
    cout << "first layer ans = " << firstLayerAns << endl;
    pair<int, float> secondLayerAns = solveSecondLayer(firstLayerAns, keyToSearch, linearModels, data.size(), isModel,
                                                       btreeMap);
    cout << "second layer ans = " << secondLayerAns.second << endl;

    cout << "threshold = " << threshold << endl;

    if (isModel[secondLayerAns.first]) {
        int positionOfKey = binarySearchBranchless<double>(data, keyToSearch, secondLayerAns.second, threshold);
        cout << "position of key = " << positionOfKey << " value = " << data[positionOfKey] << endl;
    } else {
        tree_type btree;
        vector<double> keys;
        vector<uint32_t> values;

        auto startIndex = (uint32_t) (secondLayerAns.second + errors[secondLayerAns.first].first);
        auto endIndex = (uint32_t) (secondLayerAns.second + errors[secondLayerAns.first].second);

        startIndex = (startIndex > 0) ? startIndex : 0;

        cout << "btree: startIndex, endIndex: " << startIndex << ", " << endIndex << endl;

        for (uint32_t i = startIndex; i <= endIndex; i++) {
            keys.push_back(data[i]);
            values.push_back(indices[i]);
        }

        btree_insert(btree, keys, values, 0, keys.size());

        for (uint32_t i = startIndex; i <= endIndex; i++) {
            printf("%f1000.0 ", data[i]);
            cout << (double) data[i] << " " << indices[i] << endl;
        }
        cout << "key is at position: " << (uint32_t) keyToSearch << ", " << btree_find(btree, keyToSearch) << endl;
        printf("KeyToSearch: %f1000.0", keyToSearch);
    }

    return 0;
}