#ifndef VECTOR_H
#define VECTOR_H

#include <vector>

class Vector {
private:
    std::vector<double> elements;

public:
    // Constructors
    Vector(int size, double value = 0.0);

    // Accessors
    int size() const;
    double &operator[](int index);
    const double &operator[](int index) const;

    // Operations
    Vector operator+(const Vector &other) const;
    Vector operator*(double scalar) const;
    double dot(const Vector &other) const;
};

#endif // VECTOR_H
