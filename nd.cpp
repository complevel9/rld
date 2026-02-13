#include "nd.hpp"
#include <stdarg.h>

using namespace std;

template<class T> NdArray<T>::NdArray(Vec<T> *v, uint ndim_, uint *dim_)
    : ndim(ndim_)
{
    if (ndim <= 0) sigtrap();
    size = 1;
    dim = new uint[ndim];

    for (uint i = 0; i < ndim; i++)
    {
        uint ddim = dim_[i];
        if (ddim <= 0) sigtrap();
        size *= ddim;
        dim[i] = ddim;
    }
    if (v)
    {
        v->data = (data = new T[size]);
        v->size = size;
        is_view = true;
    }
    else
    {
        data = new T[size];
        is_view = false;
    }
}

template<class T> NdArray<T>::NdArray(Vec<T> *v, uint ndim_, ...)
    : ndim(ndim_)
{
    if (ndim <= 0) sigtrap();

    va_list list;
    va_start(list, ndim_);

    size = 1;
    dim = new uint[ndim];

    for (uint i = 0; i < ndim; i++)
    {
        uint ddim = va_arg(list, uint);
        if (ddim <= 0) sigtrap();
        size *= ddim;
        dim[i] = ddim;
    }
    va_end(list);
    if (v)
    {
        v->data = (data = new T[size]);
        v->size = size;
        is_view = true;
    }
    else
    {
        data = new T[size];
        is_view = false;
    }
}

template<class T> NdArray<T>::~NdArray()
{
    delete[] dim;
    if (!is_view)
        delete[] data;
}

template<class T> T& NdArray<T>::operator[](uint *ilist) {
    uint flatpos = 0;
    for (uint i = 0; i < ndim; i++)
    {
        uint c = ilist[i];
        if (c >= dim[i])
        {
            fprintf(stderr, "index #%u is %u out of range of %u",
                    i+1, c, dim[i]);
            sigtrap();
        }
        flatpos += c;
        if (i == ndim - 1) break;
        flatpos *= dim[i+1];
    }
    return data[flatpos];
}

template<class T> T& NdArray<T>::at(uint i, ...) {
    va_list list;
    va_start(list, i);
    uint flatpos_ = flatpos(i, list);
    va_end(list);
    return data[flatpos_];
}

template<class T> uint NdArray<T>::flatpos(uint fi, ...) {
    uint flatpos = 0;
    va_list list;
    va_start(list, fi);

    for (uint i = 0; i < ndim; i++)
    {
        uint c = i == 0 ? fi : va_arg(list, uint);
        if (c >= dim[i])
        {
            fprintf(stderr, "index #%u is %u out of range of %u",
                    i+1, c, dim[i]);
            sigtrap();
        }
        flatpos += c;
        if (i == ndim - 1) break;
        flatpos *= dim[i+1];
    }
    va_end(list);
    return flatpos;
}


template<class T> void NdArray<T>::print_dim()
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

//template struct NdArray<int>;
template struct NdArray<float>;
//template struct NdArray<double>;

//template struct Vec<int>;
//template struct Vec<float>;
//template struct Vec<double>;
//template struct SparseVec<int>;
//template struct SparseVec<float>;
//template struct SparseVec<double>;


/*
template<class T> struct NdRange {
    T *a, *b;
    uint ndim;
    NdRange(uint ndim_, T *a_, T *b_);
    ~NdRange();
    NdRange(const NdArray<T>&) = delete;
    NdRange<T>& operator=(const NdRange<T>&) = delete;
    void print();
};

template<class T> NdRange<T>::NdRange(uint ndim_, T *a_, T *b_)
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

template<class T> NdRange<T>::~NdRange()
{
    delete[] a;
    delete[] b;
}

template<class T> void NdRange<T>::print()
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
