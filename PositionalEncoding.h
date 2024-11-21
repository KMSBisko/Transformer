#ifndef POSITIONAL_ENCODING_H
#define POSITIONAL_ENCODING_H

#include "Matrix.h"

class PositionalEncoding {
public:
    static Matrix compute(int seqLength, int modelDim);
};

#endif // POSITIONAL_ENCODING_H
