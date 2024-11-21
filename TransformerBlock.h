#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include "MultiHeadAttention.h"
#include "FeedForward.h"

class TransformerBlock {
public:
    MultiHeadAttention mha;
    FeedForward ff;

    TransformerBlock(int modelDim, int numHeads, int ffDim);
    Matrix forward(const Matrix &input);
};

#endif // TRANSFORMER_BLOCK_H
