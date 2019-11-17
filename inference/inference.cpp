// g++ -O3 -mavx2 -lpthread -std=c++11 -o inference inference.cpp

#include <fstream>
#include <utility>
#include <cmath>
#include <unordered_map>
#include <tuple>

#include "inference.hpp"
#include "lms_algo.h"
#include "btree.h"

using namespace std;
typedef chrono::high_resolution_clock Clock;

typedef int (*SecondLayerFun)(const double &, const double &, const uint64_t,
                              const vector<pair<double, double> > &, const double &,
                              unordered_map<int, tree_type> &, vector<uint64_t> &, int, int, vector<pair<uint32_t, uint32_t >> &);

#define NUM_ITERS 10000ll
#define NO_OF_KEYS 10000ll

template<typename T>
void print(T out) {
    for (int i = 0; i < 32; i++) {
        printf("%f1000", out.m[0][i]);
        cout << endl;
    }
    cout << endl;
}

static Mat1x16 out_1;
static Mat1x16 out_2;

void print(Mat1x16 &mat) {
    for (int i = 0; i < 16; i++) {
        printf("%f1000 ", mat.m[0][i]);
    }
    cout << endl;
}

inline
double solveFirstLayer(const Mat1x16 &hidden_layer_1, const Mat16x16 &hidden_layer_2, const Mat1x16 &output_layer,
                       const Mat1x16 &key, const Mat1x16 &bias_1, const Mat1x16 &bias_2, const double &bias_3) {

    matmult_AVX_1x1x16(out_1, key, hidden_layer_1);
    addBias(out_1, out_1, bias_1);
    relu<1, 16>(out_1);
//    print(out_1);
    matmult_AVX_1x16x16(out_2, out_1, hidden_layer_2);
    addBias(out_2, out_2, bias_2);
    relu<1, 16>(out_2);
//    print(out_2);
//    double res =

    return matmult_AVX_1x16x1_REF(out_2, output_layer) + bias_3;
}

static uint32_t midPoint;

template<bool isModel>
inline
int solveSecondLayer(const double &firstLayerOutput, const double &key, const uint64_t intKey,
                     const vector<pair<double, double> > &linearModels, const double &N,
                     unordered_map<int, tree_type> &btreeMap, vector<uint64_t> &data, int threshold,
                     int modelIndex, vector<pair<uint32_t, uint32_t >> &btreeErrors) {

    if (isModel) {
        midPoint = (key * linearModels[modelIndex].first) + linearModels[modelIndex].second;
        return binarySearchBranchless<uint64_t>(data, intKey, midPoint, threshold);
    } else {
        return binarySearchBranchless2<uint64_t>(data, intKey, btreeErrors[modelIndex].first, btreeErrors[modelIndex].second);
    }
}


tuple<vector<double>, vector<uint64_t>> getKeyList(vector<uint64_t> data, uint32_t dataLines, double maxValue) {
    vector<double> keys;
    vector<uint64_t> intKeys;
    for (int i = 0; i < NO_OF_KEYS; i++) {
        keys.push_back((data[(std::rand() % dataLines)] / ((double) 100)) / maxValue);
        intKeys.push_back(data[(std::rand() % dataLines)]);
    }
    return make_tuple(keys, intKeys);
}

tuple<vector<uint32_t>, vector<uint64_t>, double, double, uint32_t> readData(char *dataFileName) {
    ifstream dataFile(dataFileName);
    if (dataFile.is_open()) {
        int dataLines;
        uint32_t maxKey;
        uint32_t tempInt32;
        double tempDouble, offset;
        uint64_t maxValue;

        vector<uint64_t> data;
        vector<uint32_t> indices;

        dataFile >> dataLines >> maxValue;
        for (int i = 0; i < dataLines; ++i) {
            dataFile >> tempInt32;
            dataFile >> tempDouble;

            if (indices.size() == 0) {
                offset = tempDouble;
            } else {
                while (i < dataLines && tempInt32 != indices.back() + 1) {
                    indices.push_back(indices.back() + 1);
                    data.push_back(data.back());
                }
            }
            indices.push_back(tempInt32);
            data.push_back((uint64_t) ((tempDouble - offset) * 100));
        }
        dataFile.close();
        return make_tuple(indices, data, offset, (double) maxValue, dataLines);
    } else {
        cout << "Unable to open data file." << endl;
        exit(0);
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        cout << "Usage:\ninference <data file> <first layer weights file> <second layer weights>" << endl;
        exit(0);
    }

    double offset, maxValue;
    int dataLines;
    vector<uint32_t> indices;
    vector<uint64_t> data;
    double temp1, temp2, temp3;
    double tempDouble;

    tie(indices, data, offset, maxValue, dataLines) = readData(argv[1]);

    Mat1x16 hidden_layer_1;
    Mat16x16 hidden_layer_2;
    Mat1x16 output_layer, bias_1, bias_2;

    Mat1x16 key;
    double bias_3;

    ifstream firstLayerWeightsFile(argv[2]);
    if (firstLayerWeightsFile.is_open()) {
        for (int i = 0; i < 16; ++i) {
            firstLayerWeightsFile >> hidden_layer_1.m[0][i];
        }
        for (int i = 0; i < 16; ++i) {
            firstLayerWeightsFile >> bias_1.m[0][i];
        }
        for (int i = 0; i < 16; ++i) {
            for (int j = 0; j < 16; ++j) {
                firstLayerWeightsFile >> hidden_layer_2.m[j][i];
            }
        }
        for (int i = 0; i < 16; ++i) {
            firstLayerWeightsFile >> bias_2.m[0][i];
        }
        for (int i = 0; i < 16; ++i) {
            firstLayerWeightsFile >> output_layer.m[0][i];
        }
        firstLayerWeightsFile >> bias_3;
    }
    firstLayerWeightsFile.close();

    ifstream secondLayerWeightsFile(argv[3]);
    double N, modelCount;
    int threshold;
    secondLayerWeightsFile >> modelCount >> N >> temp1 >> temp2;
    threshold = (int) temp2;
    vector<pair<double, double> > linearModels;
    vector<pair<uint32_t, uint32_t > > bTreeBounds;
    vector<bool> isModel;
    unordered_map<int, tree_type> btreeMap;
    vector<SecondLayerFun> secondLayerVec;

    for (int i = 0; i < (int) modelCount; ++i) {
        secondLayerWeightsFile >> temp1 >> temp2;
        linearModels.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile >> temp1 >> temp2;
//        errors.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile >> temp1 >> temp2;
        bTreeBounds.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile >> temp3;

        isModel.push_back(temp3 != 0.0f);
        if (!isModel.back()) {
//            if (temp1 != -1.0f && temp2 != -1.0f) {
//                tree_type btree;
//                int start = (int) (temp1 - 2);
//                start = (start < 0) ? 0 : start;
//                int end = (int) (temp2 + 2);
//                btree_insert(btree, data, indices, start, end);
//                btreeMap[i] = btree;
//            }
            secondLayerVec.push_back(solveSecondLayer<false>);
        } else {
            secondLayerVec.push_back(solveSecondLayer<true>);
        }
    }
    secondLayerWeightsFile.close();

    vector<double> keyList;
    vector<uint64_t> keyListInt;
    tie(keyList, keyListInt) = getKeyList(data, dataLines, maxValue);
    uint64_t sum = 0;
    double keyToSearch;
    double firstLayerAns;
    int secondLayerAns;
    int modelIndex;
    int i, j;

    auto t1 = Clock::now();
    for (j = 0; j < NUM_ITERS; ++j) {

        for (i = 0; i < keyList.size(); ++i) {
            keyToSearch = keyList[i];
            key.m[0][0] = keyToSearch;
            firstLayerAns = solveFirstLayer(hidden_layer_1, hidden_layer_2, output_layer, key, bias_1, bias_2, bias_3);
            tempDouble = firstLayerAns * linearModels.size();
            modelIndex = (int) tempDouble;

            secondLayerAns = secondLayerVec[modelIndex](firstLayerAns, keyToSearch, keyListInt[i], linearModels,
                                                        data.size(),
                                                        btreeMap, data, threshold + 1, modelIndex, bTreeBounds);
            sum += secondLayerAns;
//            sum += firstLayerAns;
        }
//        if (!((keyList[i] == secondLayerAns) || (data[keyList[i]] == data[secondLayerAns]))) {
//            cout << "Wrong prediction!!!!!!!!!!!" << endl;
//            cout << "Actual Key: " << keyList[i] << ", Predicted Key: " << secondLayerAns << endl;
//            assert(false);
//        }

    }
    auto t2 = Clock::now();
    cout << "sum = " << sum << endl;
    std::cout << "Time: "
              << (chrono::duration<int64_t, std::nano>(t2 - t1).count() / NUM_ITERS) / NO_OF_KEYS
              << " nanoseconds" << std::endl;

    return 0;
}