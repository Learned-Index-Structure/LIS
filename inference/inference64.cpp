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

typedef int (*SecondLayerFun)(const double &, const double &, const double,
                              const vector<pair<double, double> > &, const double &,
                              unordered_map<int, tree_type> &, vector<double> &, int, int);

#define NUM_ITERS 1000ll
#define NO_OF_KEYS 10000ll

inline
float solveFirstLayer(const Mat1x32d &hidden_layer_1, const Mat32x32d &hidden_layer_2, const Mat1x32d &output_layer,
                      const Mat1x32d &key) {

    Mat1x32d out_1;
    Mat1x32d out_2;

    matmult_AVX_1x1x32d(out_1, key, hidden_layer_1);
    relu<1, 32>(out_1);
    matmult_AVX_1x32x32d(out_2, out_1, hidden_layer_2);
    relu<1, 32>(out_2);
    return matmult_AVX_1x32x1_REFd(out_2, output_layer);
}

template<bool isModel>
inline
int solveSecondLayer(const double &firstLayerOutput, const double &key, const double doubleKey,
                     const vector<pair<double, double> > &linearModels, const double &N,
                     unordered_map<int, tree_type> &btreeMap, vector<double> &data, int threshold,
                     int modelIndex) {

    int ans;
    if (isModel) {
        double temp2 = (key * linearModels[modelIndex].first) + linearModels[modelIndex].second;
        ans = binarySearchBranchless<double>(data, doubleKey, temp2, threshold);
    } else {
        ans = btree_find(btreeMap[modelIndex], doubleKey);
    }
    return ans;
}


vector<uint32_t> getKeyList(uint32_t dataLines) {
    vector<uint32_t> keys;
    for (int i = 0; i < NO_OF_KEYS; i++) {
        keys.push_back((std::rand() % dataLines));
    }
    return keys;
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
    double temp1, temp2, temp3;
    uint32_t tempInt1, tempInt2;
    double tempDouble;

    ifstream dataFile(argv[1]);
    if (dataFile.is_open()) {
        dataFile >> dataLines;
        for (int i = 0; i < dataLines; ++i) {
            dataFile >> tempInt1;
            dataFile >> tempDouble;

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

    Mat1x32d hidden_layer_1;
    Mat32x32d hidden_layer_2;
    Mat1x32d output_layer;
    Mat1x32d key;

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
    double N, modelCount;
    int threshold;
    secondLayerWeightsFile >> modelCount >> N >> temp1 >> temp2;
    threshold = (int) temp2;
    vector<pair<double, double> > linearModels;
    vector<pair<double, double> > errors;
    vector<bool> isModel;
    unordered_map<int, tree_type> btreeMap;
    vector<SecondLayerFun> secondLayerVec;

    for (int i = 0; i < (int) modelCount; ++i) {
        secondLayerWeightsFile >> temp1 >> temp2;
        linearModels.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile >> temp1 >> temp2;
        errors.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile >> temp1 >> temp2;
        secondLayerWeightsFile >> temp3;

        isModel.push_back(temp3 != 0.0f);
        if (!isModel.back()) {
            if (temp1 != -1.0f && temp2 != -1.0f) {
                tree_type btree;
                int start = (int) (temp1 - 2);
                start = (start < 0) ? 0 : start;
                int end = (int) (temp2 + 2);
                btree_insert(btree, data, indices, start, end);
                btreeMap[i] = btree;
            }
            secondLayerVec.push_back(solveSecondLayer<false>);
        } else {
            secondLayerVec.push_back(solveSecondLayer<true>);
        }
    }
    secondLayerWeightsFile.close();

    vector<uint32_t> keyList = getKeyList(dataLines);
    auto t1 = Clock::now();
    uint64_t sum = 0;
    for (int i = 0; i < keyList.size(); ++i) {
        double keyToSearch = data[keyList[i]];

        key.m[0][0] = keyToSearch;
        int secondLayerAns;
        for (int j = 0; j < NUM_ITERS; ++j) {
            float firstLayerAns = solveFirstLayer(hidden_layer_1, hidden_layer_2, output_layer, key);
            float temp = firstLayerAns * linearModels.size() / N;
            int modelIndex = (int) temp;
//            cout << "first layer ans = " << firstLayerAns << endl;
            secondLayerAns = secondLayerVec[modelIndex](firstLayerAns, keyToSearch, keyToSearch, linearModels,
                                                        data.size(),
                                                        btreeMap, data, threshold + 1, modelIndex);
            sum += secondLayerAns;
        }
        if (!((keyList[i] == secondLayerAns) || (data[keyList[i]] == data[secondLayerAns]))) {
            cout << "Wrong prediction!!!!!!!!!!!" << endl;
            cout << "Actual Key: " << keyList[i] << ", Predicted Key: " << secondLayerAns << endl;
            assert(false);
        }
//        cout << "position of key = " << secondLayerAns << " value = " << keyToSearch << endl;
//        cout<<"===========================\n\n";
    }
    auto t2 = Clock::now();
    cout << "sum - " << sum << endl;
    std::cout << "Time: "
              << (chrono::duration<int64_t, std::nano>(t2 - t1).count() / NUM_ITERS) / NO_OF_KEYS
              << " nanoseconds" << std::endl;

    return 0;
}