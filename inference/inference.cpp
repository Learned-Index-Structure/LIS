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

static Mat1x16 out_1;
static Mat1x16 out_2;

static Mat1x16 hidden_layer_1;
static Mat16x16 hidden_layer_2;
static Mat1x16 output_layer;
static Mat1x16 key;
static Mat1x16 bias_1;
static Mat1x16 bias_2;
static double bias_3;
static uint64_t multiplier = 0;

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

static uint32_t midPoint;
double maxKey;
double maxIndex;

static double firstLayerOutput;
static vector<pair<double, double> > linearModels;
static vector<uint64_t> tData;
static int threshold;
static int modelIndex;
static vector<pair<uint32_t, uint32_t >> btreeErrors;
static uint64_t keyListIntVal;

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


tuple<vector<double>, vector<uint64_t>> getKeyList(vector<uint64_t> data, uint32_t dataLines, double maxKey) {
    vector<double> keys;
    vector<uint64_t> intKeys;
//    // Random seed
//    random_device rd;
//    mt19937 gen(rd());
//    // Generate pseudo-random numbers
//    // uniformly distributed in range (0, dataLines)
//    uniform_int_distribution<> dis(0, dataLines);

    for (int i = 0; i < NO_OF_KEYS; i++) {
//        uint64_t k = dis(gen);
        uint64_t k = rand() % dataLines;
        keys.push_back(data[k] / maxKey / (double) (multiplier));
        intKeys.push_back(data[k]);
    }
//        keys.push_back(0.0001415501221045024);
//        intKeys.push_back(37912779);
    return make_tuple(keys, intKeys);
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
                offset = 0; //tempDouble;
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

int main(int argc, char **argv) {
//    if (argc != 4) {
//        cout << "Usage:\ninference <data file> <first layer weights file> <second layer weights>" << endl;
//        exit(0);
//    }

    double offset;
    int dataLines;
    vector<uint32_t> indices;
    double temp1, temp2, temp3;
    double tempDouble;

    string base = "/Users/deepak/Downloads/weights/";
//    string dataType = "weblog/";
//    multiplier = 100;
//    string dataType = "maps/";
//    multiplier = 10000000000;
    string dataType = "lognormal/";
    multiplier = 1;
    string layer1 = "model_params_layer_1.txt";
    string layer2 = "model_params_layer_2.txt";
    string dataFileName = "sorted_keys_non_repeated.csv";


    string path = base + dataType;

    string argv1 = path + dataFileName;
    string argv2 = path + layer1;
    string argv3 = path + layer2;

//    strcpy(argv[1], (path + dataFileName).c_str());
//    strcpy(argv[2], (path + layer1).c_str());
//    strcpy(argv[3], (path + layer2).c_str());

//    ifstream firstLayerWeightsFile(argv[2]);
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

//    ifstream secondLayerWeightsFile(argv[3]);
    ifstream secondLayerWeightsFile(argv3);
    double N, modelCount;
    secondLayerWeightsFile >> modelCount >> maxKey >> maxIndex >> dataLines >> threshold;
    threshold = (int) threshold + 1;
    vector<bool> isModel;
    vector<SecondLayerFun> secondLayerVec;
    double bucketSize = 0;

    for (int i = 0; i < (int) modelCount; ++i) {
        secondLayerWeightsFile >> temp1 >> temp2;
        linearModels.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile >> temp1 >> temp2;
        secondLayerWeightsFile >> temp1 >> temp2;
        temp1 = (temp1 > 2) ? temp1 - 2 : temp1;
        temp2 += 2;
        btreeErrors.push_back(make_pair(temp1, temp2));
        secondLayerWeightsFile >> bucketSize >> temp3;

        isModel.push_back(temp3 != 0.0f);
        if (!isModel.back()) {
            secondLayerVec.push_back(solveSecondLayer<false>);
        } else {
            secondLayerVec.push_back(solveSecondLayer<true>);
        }
    }
    secondLayerWeightsFile.close();

//    tie(indices, tData, offset) = readData(argv[1]);
    tie(indices, tData, offset) = readData(argv1, dataLines);

    vector<double> keyList;
    vector<uint64_t> keyListInt;
    tie(keyList, keyListInt) = getKeyList(tData, dataLines, maxKey);
    uint64_t sum = 0;
    double keyToSearch;
    int secondLayerAns;
    int i, j;

    auto t1 = Clock::now();
    for (j = 0; j < NUM_ITERS; ++j) {

        for (i = 0; i < keyList.size(); ++i) {
            keyToSearch = keyList[i];
            key.m[0][0] = keyToSearch;
            firstLayerOutput = solveFirstLayer();
            tempDouble = firstLayerOutput * linearModels.size();
            modelIndex = (int) tempDouble;

            modelIndex = (modelIndex < 0) ? 0 : ((modelIndex > (modelCount - 1)) ? (modelCount - 1) : modelIndex);

            keyListIntVal = keyListInt[i];
            secondLayerAns = secondLayerVec[modelIndex]();
            sum += secondLayerAns;

            //Accuracy Test
//            if (keyListInt[i] == tData[secondLayerAns]) continue;
//            cout << "Wrong prediction!!!!!!!!!!!" << endl;
//            cout << "Actual Key: " << keyListInt[i] << ", Predicted Key: " << secondLayerAns << endl;
//            assert(false);
        }

    }

    auto t2 = Clock::now();
    cout << "sum = " << sum << endl;
    std::cout << "Time: "
              << (chrono::duration<int64_t, std::nano>(t2 - t1).count() / NUM_ITERS) / NO_OF_KEYS
              << " nanoseconds" << std::endl;
    return 0;
}