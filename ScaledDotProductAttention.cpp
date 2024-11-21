#include "ScaledDotProductAttention.h"
#include <cmath>

Matrix ScaledDotProductAttention::compute(const Matrix &query, const Matrix &key, const Matrix &value) {
    Matrix scores = query * key.transpose();
    double scale = sqrt(query.cols());
    scores = scores * (1.0 / scale);
    Matrix attention = scores.softmax(); // Implement softmax in Matrix
    return attention * value;
}
