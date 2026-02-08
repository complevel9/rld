#include "nd.hpp"

using namespace std;

template<typename T> NdArray<T>::NdArray(uint ndim_, uint *dim_)
: ndim(ndim_)
{
    if (ndim <= 0) exit(-3);
    vol = 1;
    dim = new uint[ndim];

    for (uint i = 0; i < ndim; i++)
    {
        uint ddim = dim_[i];
        if (ddim <= 0) exit(-3);
        vol *= ddim;
        dim[i] = ddim;
    }
    data = new T[vol];
}

template<typename T> NdArray<T>::~NdArray()
{
    delete[] data;
    delete[] dim;
}

template<typename T> T& NdArray<T>::operator[](uint *ilist) {
    uint flatpos = 0;
    for (uint i = 0; i < ndim; i++)
    {
        uint c = ilist[i];
        if (c >= dim[i])
        {
            fprintf(stderr, "index #%u is %u out of range of %u",
                    i+1, c, dim[i]);
            exit(-1);
        }
        flatpos += c;
        if (i == ndim - 1) break;
        flatpos *= dim[i+1];
    }
    return data[flatpos];
}

template<typename T> void NdArray<T>::print_dim()
{
    printf("dim (");
    for (uint i = 0; i < ndim; i++)
    {
        printf("%u", dim[i]);
        if (i == ndim - 1) break;
        printf(", ");
    }
    printf(") x %lu b\n", sizeof(T));
}

template struct NdArray<int>;
template struct NdArray<float>;
template struct NdArray<double>;

//template struct Vec<int>;
template struct Vec<float>;
//template struct Vec<double>;
//template struct SparseVec<int>;
template struct SparseVec<float>;
//template struct SparseVec<double>;


/*
template<typename T> struct NdRange {
    T *a, *b;
    uint ndim;
    NdRange(uint ndim_, T *a_, T *b_);
    ~NdRange();
    NdRange(const NdArray<T>&) = delete;
    NdRange<T>& operator=(const NdRange<T>&) = delete;
    void print();
};

template<typename T> NdRange<T>::NdRange(uint ndim_, T *a_, T *b_)
    : ndim(ndim_)
{
    if (ndim <= 0) exit(-4);
    a = new T[ndim];
    b = new T[ndim];
    for (uint i = 0; i < ndim; i++)
    {
        a[i] = a_[i];
        b[i] = b_[i];
        if (a[i] > b[i]) exit(-4);
    }
}

template<typename T> NdRange<T>::~NdRange()
{
    delete[] a;
    delete[] b;
}

template<typename T> void NdRange<T>::print()
{
    printf("range ");
    for (uint i = 0; i < ndim; i++)
    {
        printf("[%d, %d]");
        if (i == ndim - 1) break;
        printf(" x ");
    }
    putchar('\n');
}

*/
