#include "PositionalEncoding.h"
#include <cmath>

Matrix PositionalEncoding::compute(int seqLength, int modelDim) {
    Matrix encoding(seqLength, modelDim);
    for (int pos = 0; pos < seqLength; ++pos) {
        for (int i = 0; i < modelDim; ++i) {
            if (i % 2 == 0)
                encoding(pos, i) = sin(pos / pow(10000, i / (double)modelDim));
            else
                encoding(pos, i) = cos(pos / pow(10000, (i - 1) / (double)modelDim));
        }
    }
    return encoding;
}
