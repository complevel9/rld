#ifndef HEADER_COMMON_HPP_
#define HEADER_COMMON_HPP_

typedef unsigned int uint;
typedef float real;

#define sigtrap() asm volatile ("int $3")// int 0x03

template<class T> T clamp(T x, T a, T b)
{
    if (x < a)
        return a;
    if (x > b)
        return b;
    return x;
}

#endif
