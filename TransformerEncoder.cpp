#include "TransformerEncoder.h"

TransformerEncoder::TransformerEncoder(int numLayers, int modelDim, int numHeads, int ffDim) {
    for (int i = 0; i < numLayers; ++i)
        layers.emplace_back(modelDim, numHeads, ffDim);
}

Matrix TransformerEncoder::forward(const Matrix &input) {
    Matrix output = input;
    for (auto &layer : layers)
        output = layer.forward(output);
    return output;
}
