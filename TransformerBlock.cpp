#include "TransformerBlock.h"

TransformerBlock::TransformerBlock(int modelDim, int numHeads, int ffDim)
    : mha(numHeads, modelDim), ff(modelDim, ffDim) {}

Matrix TransformerBlock::forward(const Matrix &input) {
    Matrix attention = mha.forward(input);
    Matrix attentionResidual = attention + input;

    Matrix feedForwardOutput = ff.forward(attentionResidual);
    return feedForwardOutput + attentionResidual;
}
