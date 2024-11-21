#include "Vector.h"
#include <stdexcept>

// Constructor
Vector::Vector(int size, double value)
    : elements(size, value) {}

// Accessors
int Vector::size() const {
    return elements.size();
}

double &Vector::operator[](int index) {
    return elements[index];
}

const double &Vector::operator[](int index) const {
    return elements[index];
}

// Addition
Vector Vector::operator+(const Vector &other) const {
    if (size() != other.size())
        throw std::runtime_error("Vector size mismatch for addition");

    Vector result(size());
    for (int i = 0; i < size(); ++i)
        result[i] = (*this)[i] + other[i];
    return result;
}

// Scalar Multiplication
Vector Vector::operator*(double scalar) const {
    Vector result(size());
    for (int i = 0; i < size(); ++i)
        result[i] = (*this)[i] * scalar;
    return result;
}

// Dot Product
double Vector::dot(const Vector &other) const {
    if (size() != other.size())
        throw std::runtime_error("Vector size mismatch for dot product");

    double result = 0.0;
    for (int i = 0; i < size(); ++i)
        result += (*this)[i] * other[i];
    return result;
}
