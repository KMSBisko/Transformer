#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>

class Matrix {
private:
    std::vector<std::vector<double>> elements;

public:
    // Constructors
    Matrix(int rows, int cols, double value = 0.0);

    // Accessors
    int rows() const;
    int cols() const;
    double &operator()(int row, int col);
    const double &operator()(int row, int col) const;

    // Operations
    Matrix operator+(const Matrix &other) const;
    Matrix operator*(const Matrix &other) const;
    Matrix operator*(double scalar) const;
    
    // Utility
    Matrix transpose() const;
    Matrix relu() const;
    Matrix softmax() const;
};

#endif // MATRIX_H
