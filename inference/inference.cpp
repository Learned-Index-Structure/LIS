// g++ -O3 -mavx2 -lpthread -std=c++11 -o inference inference.cpp

#include <fstream>
#include <utility>
#include <cmath>
#include <unordered_map>
#include <tuple>
#include <random>

#include "inference.hpp"
#include "lms_algo.hpp"
#include "btree.hpp"

using namespace std;
typedef chrono::high_resolution_clock Clock;

typedef int (*SecondLayerFun)();

#define NUM_ITERS 1ll
#define NO_OF_KEYS 1000000ll

Mat1x16 out_1;
Mat1x16 out_2;

Mat1x16 hidden_layer_1;
Mat16x16 hidden_layer_2;
Mat1x16 output_layer;
Mat1x16 key;
Mat1x16 bias_1;
Mat1x16 bias_2;
double bias_3;
uint64_t multiplier = 0;

static uint32_t midPoint;
double maxKey;
double maxIndex;

double firstLayerOutput;
vector<pair<double, double> > linearModels;
vector<uint64_t> tData;
int threshold;
int modelIndex;
vector<pair<uint32_t, uint32_t >> btreeErrors;
uint64_t keyListIntVal;


string argv1;
string argv2;
string argv3;

double offset;
int dataLines;
vector<uint32_t> indices;
int modelCount;
vector<bool> isModel;
vector<SecondLayerFun> secondLayerVec;
vector<double> keyList;
vector<uint64_t> keyListInt;

inline void cleanup() {
    indices.clear();
    isModel.clear();
    secondLayerVec.clear();
    linearModels.clear();
    tData.clear();
    btreeErrors.clear();
    keyList.clear();
    keyListInt.clear();
}

inline
double solveFirstLayer() {
    matmult_AVX_1x1x16(out_1, key, hidden_layer_1);
    addBias(out_1, out_1, bias_1);
    relu<1, 16>(out_1);
    matmult_AVX_1x16x16(out_2, out_1, hidden_layer_2);
    addBias(out_2, out_2, bias_2);
    relu<1, 16>(out_2);

    return matmult_AVX_1x16x1_REF(out_2, output_layer) + bias_3;
}


template<bool isModel>
inline
int solveSecondLayer() {
    if (isModel) {
        midPoint = (uint32_t) (((key.m[0][0] * linearModels[modelIndex].first) + linearModels[modelIndex].second) *
                               maxIndex);
        return binarySearchBranchless<uint64_t>(tData, keyListIntVal, midPoint, threshold);
    } else {
        return binarySearchBranchless2<uint64_t>(tData, keyListIntVal, btreeErrors[modelIndex].first,
                                                 btreeErrors[modelIndex].second);
    }
}


inline void getKeyList(vector<uint64_t> data, uint32_t dataLines, double maxKey) {
    // Random seed
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, dataLines);

    for (int i = 0; i < NO_OF_KEYS; i++) {
        uint64_t k = dis(gen);
        keyList.push_back(data[k] / maxKey / (double) (multiplier));
        keyListInt.push_back(data[k]);
    }
//        keys.push_back(0.0001415501221045024);
//        intKeys.push_back(37912779);
}

tuple<vector<uint32_t>, vector<uint64_t>, double> readData(string dataFileName, uint32_t dataLines) {
    ifstream dataFile(dataFileName);
    if (dataFile.is_open()) {

        uint32_t tempInt32;
        double tempDouble, offset;

        vector<uint64_t> data;
        vector<uint32_t> indices;

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
            data.push_back((uint64_t) ((tempDouble - offset) * multiplier));
        }
        dataFile.close();
        return make_tuple(indices, data, offset);
    } else {
        cout << "Unable to open data file." << endl;
        exit(0);
    }
}


void setup(string basePath, string dataset, string modelCountStr, string thresholdStr) {

    // Filename Setup
    if (dataset.compare("weblog"))
        multiplier = 100;
    else if (dataset.compare("maps"))
        multiplier = 10000000000;
    else
        multiplier = 1;

    dataset = dataset + "/";
    string layer1 = "model_params_layer_1.txt";
    string layer2 = "model_params_layer_2";

//    layer1 = layer1 + "_" + modelCountStr + "_" + thresholdStr + ".txt";
    layer2 = layer2 + "_" + modelCountStr + "_" + thresholdStr + ".txt";

    string dataFileName = "sorted_keys_non_repeated.csv";

    string path = basePath + dataset;

    argv1 = path + dataFileName;
    argv2 = path + layer1;
    argv3 = path + layer2;

    //Load data
    double temp1, temp2, temp3;

    ifstream firstLayerWeightsFile(argv2);
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

    ifstream secondLayerWeightsFile(argv3);
    secondLayerWeightsFile >> modelCount >> maxKey >> maxIndex >> dataLines >> threshold;
    threshold = (int) threshold + 1;

    double bucketSize = 0;

    for (int i = 0; i < modelCount; ++i) {
        secondLayerWeightsFile >> temp1 >> temp2;
        linearModels.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile >> temp1 >> temp2;
        secondLayerWeightsFile >> temp1 >> temp2;
        temp1 = (temp1 > 2) ? temp1 - 2 : temp1;
        temp2 += 2;
        btreeErrors.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile >> bucketSize >> temp3;
//        secondLayerWeightsFile >> temp3;

        isModel.push_back(temp3 != 0.0f);
        if (!isModel.back()) {
            secondLayerVec.push_back(solveSecondLayer<false>);
        } else {
            secondLayerVec.push_back(solveSecondLayer<true>);
        }
    }
    secondLayerWeightsFile.close();
    cout << "File Read starts." << endl;
    tie(indices, tData, offset) = readData(argv1, dataLines);
    cout << "File Read ends. File Size: " << tData.size() << endl;
}

inline uint32_t infer(uint64_t keyInt) {
    firstLayerOutput = solveFirstLayer();
    modelIndex = (int) (firstLayerOutput * linearModels.size());
    modelIndex = (modelIndex < 0) ? 0 : ((modelIndex > (modelCount - 1)) ? (modelCount - 1) : modelIndex);
    keyListIntVal = keyInt;
    return secondLayerVec[modelIndex]();
}
//
//int main(int argc, char **argv) {
//    string path = "/Users/deepak/Downloads/weights/";
//    setup(path, "weblog", "100000", "128");
//
//    vector<double> keyList;
//    vector<uint64_t> keyListInt;
//    tie(keyList, keyListInt) = getKeyList(tData, dataLines, maxKey);
//    uint64_t sum = 0;
//    double keyToSearch;
//    int i, j;
//
//    auto t1 = Clock::now();
//    for (j = 0; j < NUM_ITERS; ++j) {
//        for (i = 0; i < keyList.size(); ++i) {
//            keyToSearch = keyList[i];
//            key.m[0][0] = keyToSearch;
//            sum += infer(keyListInt[i]);
//        }
//    }
//
//    auto t2 = Clock::now();
//    cout << "sum = " << sum << endl;
//    std::cout << "Time: "
//              << (chrono::duration<int64_t, std::nano>(t2 - t1).count() / NUM_ITERS) / NO_OF_KEYS
//              << " nanoseconds" << std::endl;
//    return 0;
//}