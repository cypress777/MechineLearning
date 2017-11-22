//
// Created by cypress on 17/11/2017.
//

#ifndef ML_MFNN_HPP
#define ML_MFNN_HPP

#include <vector>

using namespace std;

class MFNN {
private:
    int m_inputSize;
    int m_hiddenLayerSize;
    int m_outputLayerSize;

    vector<double> m_hiddenLayerBias;
    vector<vector<double>> m_weightIH;
    vector<vector<double>> m_weightHO;
    vector<double > m_outputLayerBias;

    vector<double> m_hiddenLayerOutput;
    vector<double> m_outputLayerOutput;
    double m_learnRate;

    double fSigmoid(double);
    double calLayersOutput(const vector<double>&, const vector<double>&);

public:
    MFNN(int, int, int, double);

    double trainbyStdBP(const vector<double >&, const vector<double>&);

    double examTrainResult(const vector<double >&, const vector<double>&);

    void showParaMeters();
};


#endif //ML_MFNN_HPP
