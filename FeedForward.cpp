#include "FeedForward.h"

FeedForward::FeedForward(int modelDim, int ffDim)
    : W1(modelDim, ffDim), W2(ffDim, modelDim) {}

Matrix FeedForward::forward(const Matrix &input) {
    Matrix hidden = input * W1;
    hidden = hidden.relu(); // Implement ReLU in Matrix
    return hidden * W2;
}
