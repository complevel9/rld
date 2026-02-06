#define PPCAT_NOEX(A, B) A##B
#define PPCAT(A, B) PPCAT_NOEX(A,B)

// generic type, can be float/double/whatever
#define nd_dtype PPCAT(nd_, dtype)

typedef struct {
    dtype *data;
    uint *dim;
    uint vol;
    uint ndim;
} nd_dtype;

nd_dtype PPCAT(create_nd_, dtype)(uint ndim, ...);
void PPCAT(free_nd_, dtype)(nd_dtype f);
uint PPCAT(vol_nd_, dtype)(nd_dtype f);

void PPCAT(print_dim_, dtype)(nd_dtype f);

#ifdef ND_ARRAY_IMPL

// create
// creates new n-d array from arguments
nd_dtype PPCAT(create_from_arg_nd_, dtype)(uint ndim, ...)
{
    nd_dtype f;
    uint vol = 1;
    va_list list;

    f.ndim = ndim;
    f.dim = malloc(ndim * sizeof f.dim[0]);

    va_start(list, ndim);
    for (uint i = 0; i < ndim; i++)
    {
        uint ddim = va_arg(list, uint);
        if (ddim <= 0) exit(-2);
        vol *= ddim;
        f.dim[i] = ddim;
    }
    va_end(list);

    f.vol = vol;
    f.data = malloc(vol * sizeof f.data[0]);
    return f;
}

// create_from_dim
// create new n-d array from existing dim array, copies the dim over
nd_dtype PPCAT(create_from_dim_nd_, dtype)(uint ndim, uint *dim)
{
    nd_dtype f;
    uint vol = 1;
    f.ndim = ndim;
    f.dim = malloc(ndim * sizeof f.dim[0]);

    for (uint i = 0; i < ndim; i++)
    {
        uint ddim = dim[i];
        if (ddim <= 0) exit(-3);
        vol *= ddim;
        f.dim[i] = ddim;
    }
    f.vol = vol;
    f.data = malloc(vol * sizeof f.data[0]);
    return f;
}

// vol
extern inline uint PPCAT(vol_nd_, dtype)(nd_dtype f)
{
    uint vol = 1;
    for (uint i = 0; i < f.ndim; i++)
        vol *= f.dim[i];
    return vol;
}

// print_dim
void PPCAT(print_dim_, dtype)(nd_dtype f)
{
    printf("shape (");
    for (uint i = 0; i < f.ndim; i++)
    {
        printf("%d", f.dim[i]);
        if (i == f.ndim - 1) break;
        printf(", ");
    }
    printf(")\n");
}

#endif
