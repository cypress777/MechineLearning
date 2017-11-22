//
// Created by cypress on 17/11/2017.
//
#include "MFNN.hpp"
#include <cmath>
#include <random>
#include <iostream>

using namespace std;

MFNN::MFNN(int in, int hidden, int out, double lr):m_inputSize(in), m_hiddenLayerSize(hidden), m_outputLayerSize(out), m_learnRate(lr){
    default_random_engine e;
    uniform_real_distribution<double> u(0, 1);

    for (int i = 0; i < hidden; i++) {
        m_hiddenLayerBias.push_back(u(e));

        vector<double> ih;
        for (int j = 0; j < in; j++)
            ih.push_back(u(e));
        m_weightIH.push_back(ih);
    }

    for (int i = 0; i < out; i++) {
        m_outputLayerBias.push_back(u(e));

        vector<double> ho;
        for (int k = 0; k < hidden; k++)
            ho.push_back(u(e));
        m_weightHO.push_back(ho);
    }
}

double MFNN::trainbyStdBP(const vector<double>& inputLayer, const vector<double>& label) {
    double err = calLayersOutput(inputLayer, label);

    vector<double> g;
    for (int i = 0; i < m_outputLayerSize; i++) {
        double yi = m_outputLayerOutput[i];
        g.push_back(yi * (1.0 - yi) * (label[i] - yi));
    }

    vector<double> e;
    for (int i = 0; i < m_hiddenLayerSize; i++) {
        double bi = m_hiddenLayerOutput[i];
        double sum = 0;
        for (int j = 0; j < m_outputLayerSize; j++) {
            sum += m_weightHO[j][i] * g[j];
        }
        e.push_back(bi * (1.0 - bi) * sum);
    }

    for (int i = 0; i < m_outputLayerSize; i++) {
        for (int j = 0; j < m_hiddenLayerSize; j++)
            m_weightHO[i][j] += (m_learnRate * g[i] * m_hiddenLayerOutput[j]);
    }

    for (int i = 0; i < m_outputLayerSize; i++)
        m_outputLayerBias[i] += (-1.0 * m_learnRate * g[i]);

    for (int i = 0; i < m_hiddenLayerSize; i++) {
        for (int j = 0; j < m_inputSize; j++)
            m_weightIH[i][j] += (m_learnRate * e[i] * inputLayer[j]);
    }

    for (int i = 0; i < m_hiddenLayerSize; i++)
        m_hiddenLayerBias[i] += (-1.0 * m_learnRate * e[i]);

    return err;
}

double MFNN::examTrainResult(const vector<double >& inputLayer, const vector<double>& label) {
    double err = calLayersOutput(inputLayer, label);
    cout << "-----------result------------" << endl;

    cout << "input: " << endl;
    for (auto item : inputLayer)
        cout << item << "  ";
    cout << endl;

    for (int i = 0; i < m_outputLayerSize; i++) {
        cout << "cal: " << m_outputLayerOutput[i] << " true: " << label[i] << endl;
    }
    cout << "error: " << err << endl;

    return err;
}

void MFNN::showParaMeters(){

    cout << "------------output bias: " << endl;
    for (int i = 0; i < m_outputLayerSize; i++) {
        cout << i << ": " << m_outputLayerBias[i] << "  ";
    }
    cout << endl;

    cout << "------------HO weight: " << endl;
    for (int i = 0; i < m_outputLayerSize; i++) {
        cout << i << ": ";
        for (int j = 0; j < m_hiddenLayerSize; j++) {
            cout << m_weightHO[i][j] << "  ";
        }
        cout << endl;
    }


    cout << "------------hidden bias: " << endl;
    for (int i = 0; i < m_hiddenLayerSize; i++) {
        cout << m_hiddenLayerBias[i] << "  ";
    }
    cout << endl;

    cout << "------------IH weight: " << endl;
    for (int i = 0; i < m_hiddenLayerSize; i++) {
        cout << i <<  ": ";
        for (int j = 0; j < m_inputSize; j++) {
            cout  << m_weightIH[i][j] << "  ";
        }
        cout << endl;
    }

}

double MFNN::calLayersOutput(const vector<double>& inputLayer, const vector<double>& label) {
    m_hiddenLayerOutput.erase(m_hiddenLayerOutput.begin(), m_hiddenLayerOutput.end());
    m_outputLayerOutput.erase(m_outputLayerOutput.begin(), m_outputLayerOutput.end());

    double err = 0.0;

    for (int i = 0; i < m_hiddenLayerSize; i++) {
        double inputSum = 0;
        for (int j = 0; j < m_inputSize; j++) {
            inputSum += (inputLayer[j] * m_weightIH[i][j]);
        }
        double output = fSigmoid(inputSum - m_hiddenLayerBias[i]);
        m_hiddenLayerOutput.push_back(output);
    }

    for (int i = 0; i < m_outputLayerSize; i++) {
        double inputSum = 0;
        for (int j = 0; j < m_hiddenLayerSize; j++) {
            inputSum += (m_hiddenLayerOutput[j] * m_weightHO[i][j]);
        }
        double output = fSigmoid(inputSum - m_outputLayerBias[i]);
        m_outputLayerOutput.push_back(output);
        err += (label[i] - output) * (label[i] - output);
    }
    err /= 2.0;
    return (err);
}

double MFNN::fSigmoid(double x) {
    return 1.0/(1.0 + exp(-1.0 * x));
}
