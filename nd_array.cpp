#define ND_ARRAY_IMPL
#include "nd_array.h"

extern inline uint nd_pos_fromdim(uint ndim, uint *dim, ...)
{
    size_t pos = 0;
    va_list list;
    va_start(list, dim);

    for (int i = 0; i < ndim; i++)
    {
        uint c = va_arg(list, uint);
        if (c >= dim[i])
        {
            fprintf(stderr, "index #%d is %d out of range of %d", i+1, c, dim[i]);
            exit(-1);
        }
        pos += c;
        if (i == ndim - 1) break;
        pos *= dim[i+1];
    }
    va_end(list);
    return pos;
}
