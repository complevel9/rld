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
    void set_all(T v)
    {
        for (uint i = 0; i < size; i++)
            data[i] = v;
    }
    void print()
    {
        for (uint i = 0; i < size; i++)
            printf("(%.2f) ", data[i]);
        putchar('\n');
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

typedef struct SVEntry_s {
    uint idx;
    float val;
} SVEntry;
typedef struct SparseVec_s {
    SVEntry *entries;
    uint nentries;
    uint cap;
} SparseVec;

typedef struct Discrete_s {
    uint n;
} Discrete; // { 0...n-1 }
typedef struct Range_s {
    real a, b;
} Range; // [a, b]

typedef union SimpleSet_s {
    Discrete discrete;
    Range range;
} SimpleSet;
typedef union SimpleElem_s {
    uint i;
    real r;
} SimpleElem;

// cartesian product of simple sets
typedef struct Space_s {
    SimpleSet *factors;
    uint nfactors;
} Space;
typedef struct Elem_s {
    SimpleElem *components;
    Space *space;
    uint ncomponents;
} Elem;

typedef enum OOBBehavior_e {
    OOBB_CLAMP, OOBB_LOOP, OOBB_DIE
} OOBBehavior;
typedef struct UniformRangeSplitAggregator_s {
     OOBBehavior oob_behavior;
     real a, b;
     uint nsplits;
} UniformRangeSplitAggregator;

typedef struct AggregatorSequenceFeatures_s {
} AggregatorSequenceFeatures;
/*
mcsfeatures = aggregator sequence features ( [uniform range splitting(), ] )
vec theta = vec(mcfeatures.dim)
for a in env.actions(s)
    real value = mcfapprox.get(mcfeatures.apply([...s, ...a]))

vec grad = mcfapprox.get()

*/
//extern template struct NdArray<int>;
extern template struct NdArray<float>;
//extern template struct NdArray<double>;
//extern template struct Vec<float>;
//extern template struct SparseVec<float>;

#endif
