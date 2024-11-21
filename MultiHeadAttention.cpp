#include "MultiHeadAttention.h"

MultiHeadAttention::MultiHeadAttention(int numHeads, int modelDim)
    : numHeads(numHeads), modelDim(modelDim), headDim(modelDim / numHeads),
      Wq(modelDim, modelDim), Wk(modelDim, modelDim),
      Wv(modelDim, modelDim), Wo(modelDim, modelDim) {}

Matrix MultiHeadAttention::forward(const Matrix &input) {
    Matrix query = input * Wq;
    Matrix key = input * Wk;
    Matrix value = input * Wv;

    return ScaledDotProductAttention::compute(query, key, value) * Wo;
}
