#ifndef MULTI_HEAD_ATTENTION_H
#define MULTI_HEAD_ATTENTION_H

#include "Matrix.h"
#include "ScaledDotProductAttention.h"

class MultiHeadAttention {
public:
    int numHeads, modelDim, headDim;
    Matrix Wq, Wk, Wv, Wo;

    MultiHeadAttention(int numHeads, int modelDim);
    Matrix forward(const Matrix &input);
};

#endif // MULTI_HEAD_ATTENTION_H
