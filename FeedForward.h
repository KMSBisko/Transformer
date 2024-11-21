#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include "Matrix.h"

class FeedForward {
public:
    Matrix W1, W2;
    FeedForward(int modelDim, int ffDim);
    Matrix forward(const Matrix &input);
};

#endif // FEED_FORWARD_H
