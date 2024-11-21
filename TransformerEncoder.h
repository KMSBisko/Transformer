#ifndef TRANSFORMER_ENCODER_H
#define TRANSFORMER_ENCODER_H

#include "TransformerBlock.h"
#include <vector>

class TransformerEncoder {
public:
    std::vector<TransformerBlock> layers;
    
    TransformerEncoder(int numLayers, int modelDim, int numHeads, int ffDim);
    Matrix forward(const Matrix &input);
};

#endif // TRANSFORMER_ENCODER_H
