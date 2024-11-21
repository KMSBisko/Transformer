#include "Matrix.h"
#include <cmath>

// Constructor
Matrix::Matrix(int rows, int cols, double value)
    : elements(rows, std::vector<double>(cols, value)) {}

// Accessors
int Matrix::rows() const {
    return elements.size();
}

int Matrix::cols() const {
    return elements[0].size();
}

double &Matrix::operator()(int row, int col) {
    return elements[row][col];
}

const double &Matrix::operator()(int row, int col) const {
    return elements[row][col];
}

// Addition
Matrix Matrix::operator+(const Matrix &other) const {
    int max_rows = std::max(rows(), other.rows());
    int max_cols = std::max(cols(), other.cols());

    // Create padded matrices
    Matrix padded_this(max_rows, max_cols, 0.0);
    Matrix padded_other(max_rows, max_cols, 0.0);

    // Copy original matrices into padded matrices
    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            padded_this(i, j) = (*this)(i, j);

    for (int i = 0; i < other.rows(); ++i)
        for (int j = 0; j < other.cols(); ++j)
            padded_other(i, j) = other(i, j);

    // Perform addition on padded matrices
    Matrix result(max_rows, max_cols);
    for (int i = 0; i < max_rows; ++i)
        for (int j = 0; j < max_cols; ++j)
            result(i, j) = padded_this(i, j) + padded_other(i, j);

    return result;
}

// Multiplication
Matrix Matrix::operator*(const Matrix &other) const {
    int max_size = std::max(cols(), other.rows());

    // Create padded matrices
    Matrix padded_this(rows(), max_size, 0.0);
    Matrix padded_other(max_size, other.cols(), 0.0);

    // Copy original matrices into padded matrices
    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            padded_this(i, j) = (*this)(i, j);

    for (int i = 0; i < other.rows(); ++i)
        for (int j = 0; j < other.cols(); ++j)
            padded_other(i, j) = other(i, j);

    // Perform multiplication on padded matrices
    Matrix result(rows(), other.cols(), 0.0);
    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < other.cols(); ++j)
            for (int k = 0; k < max_size; ++k)
                result(i, j) += padded_this(i, k) * padded_other(k, j);

    return result;
}

// Scalar Multiplication
Matrix Matrix::operator*(double scalar) const {
    Matrix result(rows(), cols());
    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            result(i, j) = (*this)(i, j) * scalar;
    return result;
}

// Transpose
Matrix Matrix::transpose() const {
    Matrix result(cols(), rows());
    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            result(j, i) = (*this)(i, j);
    return result;
}

// ReLU
Matrix Matrix::relu() const {
    Matrix result = *this;
    for (int i = 0; i < rows(); ++i)
        for (int j = 0; j < cols(); ++j)
            result(i, j) = std::max(0.0, (*this)(i, j));
    return result;
}

// Softmax
Matrix Matrix::softmax() const {
    Matrix result(rows(), cols());
    for (int i = 0; i < rows(); ++i) {
        double sum = 0.0;
        for (int j = 0; j < cols(); ++j)
            sum += std::exp((*this)(i, j));

        for (int j = 0; j < cols(); ++j)
            result(i, j) = std::exp((*this)(i, j)) / sum;
    }
    return result;
}
