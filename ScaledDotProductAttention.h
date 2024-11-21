#ifndef SCALED_DOT_PRODUCT_ATTENTION_H
#define SCALED_DOT_PRODUCT_ATTENTION_H

#include "Matrix.h"

class ScaledDotProductAttention {
public:
    static Matrix compute(const Matrix &query, const Matrix &key, const Matrix &value);
};

#endif // SCALED_DOT_PRODUCT_ATTENTION_H
