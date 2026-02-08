#ifndef HEADER_ND_ARRAY_H_
#define HEADER_ND_ARRAY_H_

#include "commontypes.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cblas.h>

template<typename T> struct NdArray {
    T *data;
    uint *dim;
    uint ndim;
    uint vol;

    NdArray(uint ndim_, uint *dim_);
    ~NdArray();
    NdArray(const NdArray<T>&) = delete;
    NdArray<T>& operator=(const NdArray<T>&) = delete;
    T& operator[](uint *ilist);
    void print_dim();
};

/*
struct Set {
    enum SetType {
        finite, range, cartesian_product
    } type;
};

struct Elem {
    Set *parent;
};

struct Finite : Set {
    uint n;
};

struct Range : Set {
    float a, b;
    Range(float a, float b);
    void print();
};

struct CartesianProduct : Set {
    std::vector<Set> *prod;
    void print();
};
*/

template<typename T> struct Vec {
    T *data;
    uint dim;
    T lazyscale;
};


template<typename T> struct SparseVec {
    uint *idx;
    T *data;
    uint dim;
    uint num_entries;
};


//extern template struct NdArray<int>;
extern template struct NdArray<float>;
//extern template struct NdArray<double>;
extern template struct Vec<float>;
extern template struct SparseVec<float>;

#endif
