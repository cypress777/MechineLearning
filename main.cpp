#include <vector>
#include <iostream>
#include <cmath>
#include "main.hpp"
#include "MFNN.hpp"

using namespace std;

vector<MelonInfo> Melons {
        {green, wave, dull, clear, dent, hard, 0.697, 0.460, true},
        {black, wave, heavy, clear, dent, hard, 0.774, 0.376, true},
        {black, wave, dull, clear, dent, hard, 0.634, 0.264, true},
        {green, wave, heavy, clear, dent, hard, 0.608, 0.318, true},
        {white, wave, dull, clear, dent, hard, 0.556, 0.215, true},
        {green, curl, dull, clear, slight, sticky, 0.403, 0.237, true},
        {black, curl, dull, clear, dent, sticky, 0.481, 0.149, true},
        {black, curl, dull, clear, slight, hard, 0.437, 0.211, true},
        {black, curl, heavy, unclear, flat, hard, 0.666, 0.091, false},
        {green, straight, crispy, clear, flat, sticky, 0.243, 0.267, false},
        {white, straight, crispy, blur, flat, hard, 0.245, 0.057, false},
        {white, wave, heavy, blur, flat, sticky, 0.343, 0.099, false},
        {green, curl, heavy, unclear, dent, hard, 0.639, 0.161, false},
        {white, curl, dull, unclear, dent, hard, 0.657, 0.198, false},
        {black, curl, heavy, blur, slight, sticky, 0.360, 0.370, false},
        {white, wave, heavy, blur, flat, hard, 0.593, 0.042, false},
        {green, wave, dull, unclear, slight, hard, 0.719, 0.103, false},
};

vector<double> transMelonInfos(const MelonInfo& melonStruct) {
    vector<double> ret;
    ret.push_back(static_cast<double>(melonStruct.clr) - 1);
    ret.push_back(static_cast<double>(melonStruct.stm) - 1);
    ret.push_back(static_cast<double>(melonStruct.snd) - 1);
    ret.push_back(static_cast<double>(melonStruct.txtr) - 1);
    ret.push_back(static_cast<double>(melonStruct.blly) - 1);
    ret.push_back(static_cast<double>(melonStruct.feel) - 1);
    ret.push_back(static_cast<double>(melonStruct.density));
    ret.push_back(static_cast<double>(melonStruct.sugar));

    return ret;
}

void initData (vector<pair<vector<double>, vector<double>>>& melonsInfos) {
    for (const auto& melon : Melons) {
        pair<vector<double>, vector<double>> melonInfos;

        melonInfos.first = transMelonInfos(melon);

        if (melon.quality) {
            melonInfos.second.push_back(1.0);
            melonInfos.second.push_back(0.0);
        } else {
            melonInfos.second.push_back(0.0);
            melonInfos.second.push_back(1.0);
        }

        melonsInfos.push_back(melonInfos);
    }
}

int main() {
    vector<pair<vector<double>, vector<double>>> melonsInfos;
    initData(melonsInfos);

    MFNN testNN{static_cast<int>(melonsInfos[0].first.size()) , 9, static_cast<int>(melonsInfos[0].second.size()), 0.3};

    int t;
    cout << "iteration time: " << endl;
    cin >> t;

    double ferr, err;
    int iter = 0;

    for (int i = 0; i < t; i++) {
        for (const auto &meloninfos : melonsInfos) {
            vector<double> input = meloninfos.first;
            vector<double> label = meloninfos.second;

            err = testNN.trainbyStdBP(input, label);
        }
        if (abs(ferr - err) < 0.0001) {
            iter++;
            if (iter > 100) {
                cout << "stopped after " << iter <<  " iteration." <<endl;
                break;
            }
        } else {
            iter = 0;
            ferr = err;
        }
    }



    testNN.showParaMeters();

    for (const auto &meloninfos : melonsInfos) {
        vector<double> input = meloninfos.first;
        vector<double> label = meloninfos.second;

        err = testNN.examTrainResult(input, label);
    }

    cout << "completed" << endl;
    return 0;
}

