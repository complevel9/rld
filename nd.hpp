#ifndef HEADER_ND_ARRAY_HPP_
#define HEADER_ND_ARRAY_HPP_

#include "common.hpp"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#define DELETE_COPY_CTOR_ASSIGN_T(a, b) \
    a(const a<b>&) = delete;\
    a<b>& operator=(const a<b>&) = delete;

template<class T> struct Vec {
    uint size;
    T *data;
    T& operator[](uint i) { return data[i]; }
    Vec() {}
    Vec(uint size_) : size(size_), data(new T[size_]()) {}
    ~Vec() {delete[] data;}
    DELETE_COPY_CTOR_ASSIGN_T(Vec,T)
    void zero()
    {
        memset(data, 0, size * sizeof(T));
    }
    // cblasable
    void scale(T a)
    {
        for (uint i = 0; i < size; i++)
            data[i] *= a;
    }
    void add_scaled(T a, Vec<T> *v)
    {
        for (uint i = 0; i < size; i++)
            data[i] += a*v->data[i];
    }
    void add(Vec<T> *v)
    {
        for (uint i = 0; i < size; i++)
            data[i] += v->data[i];
    }
    void set_all(T v) {
        for (uint i = 0; i < size; i++)
            data[i] = v;
    }
};

template<class T> struct NdArray {
    T *data;
    uint *dim;
    uint ndim;
    uint size;
    bool is_view;

    NdArray(Vec<T> *v, uint ndim_, uint *dim_);
    NdArray(Vec<T> *v, uint ndim_, ...);
    ~NdArray();
    DELETE_COPY_CTOR_ASSIGN_T(NdArray,T)
    uint flatpos(uint i, ...);
    T& operator[](uint *ilist);
    T& at(uint i, ...);
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



template<class T> struct SparseVec {
    uint *idx;
    T *data;
    uint dim;
    uint num_entries;
};


//extern template struct NdArray<int>;
extern template struct NdArray<float>;
//extern template struct NdArray<double>;
//extern template struct Vec<float>;
//extern template struct SparseVec<float>;

#endif
