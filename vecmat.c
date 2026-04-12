// -------------------- Vec and Mat ---------------------------

// OpenBLAS for some faster matrix/vec math
#if 1
    #include <cblas.h>
    #define HAS_CBLAS 1
    #if REAL_SINGLE
        #define cblas_ger  cblas_sger
        #define cblas_syr  cblas_ssyr
        #define cblas_dot  cblas_sdot
        #define cblas_axpy cblas_saxpy
        #define cblas_scal cblas_sscal
        #define cblas_copy cblas_scopy
        #define cblas_gemv cblas_sgemv
        #define cblas_symv cblas_ssymv
    #else
        #define cblas_ger  cblas_dger
        #define cblas_syr  cblas_dsyr
        #define cblas_dot  cblas_ddot
        #define cblas_axpy cblas_daxpy
        #define cblas_scal cblas_dscal
        #define cblas_copy cblas_dcopy
        #define cblas_gemv cblas_dgemv
        #define cblas_symv cblas_dsymv
    #endif

    const char *openblas_parallel_str() {
        int openblas_parallel = openblas_get_parallel();
        switch (openblas_parallel) {
        case OPENBLAS_SEQUENTIAL:
            return "sequential";
        case OPENBLAS_THREAD:
            return "pthread";
        case OPENBLAS_OPENMP:
            return "openmp";
        }
        return "unknown";
    }
#else
    #define HAS_CBLAS 0
#endif

typedef struct {
    uint idx;
    real val;
} SparseEntry;
// set sparse cap to 0 if dense
// if sparse, assume entries are sorted by index and no duplicates
// sparse representation not to save memory but rather to save computation
typedef struct {
    union {
        real *dense;
        SparseEntry *sparse;
    } data;
    uint dim;
    ushort sparse_nentries, sparse_cap;
} Vec;

// c order matrix, same sparse rep structure
typedef struct {
    union {
        real *dense;
        SparseEntry *sparse;
    } data;
    uint nrows, ncols;
    ushort sparse_nentries, sparse_cap;
} Mat;

#define VEC_IS_SPARSE(v) ((v).sparse_cap != 0)
#define VEC_IS_DENSE(v) (!(v).sparse_cap)
#define VEC_PTR_IS_SPARSE(v) ((v)->sparse_cap != 0)
#define VEC_PTR_IS_DENSE(v) (!(v)->sparse_cap)
#define VEC_NOT_ALLOCD(v) (!(v)->data.dense)

void make_vec(Vec *v, uint dim, ushort sparse_cap) {
    // assert(dim); // dim = 0 means no allocation
    if (!dim) {
        v->data.dense = NULL;
        return;
    }
    v->dim = dim;
    v->sparse_nentries = 0;
    v->sparse_cap = sparse_cap;
    if (!sparse_cap) {
        v->data.dense = custom_malloc(dim * sizeof(real));
    } else {
        assert(sparse_cap <= dim);
        v->data.sparse = custom_malloc(sparse_cap * sizeof(SparseEntry));
    }
}

extern inline void free_vec(Vec *v) {free(v->data.dense);}

void push_entry_svec(Vec *v, uint idx, real val) {
    assert(idx < v->dim);
    assert(v->sparse_cap >= v->sparse_nentries + 1);
    ushort ie = v->sparse_nentries++;
    v->data.sparse[ie].idx = idx;
    v->data.sparse[ie].val = val;
}

void copy_vec_to_vec(Vec *v, Vec *dest) {
    assert(!VEC_NOT_ALLOCD(v));
    if (VEC_NOT_ALLOCD(dest))
        make_vec(dest, v->dim, v->sparse_nentries);
    assert(v->dim == dest->dim);
    if (VEC_PTR_IS_SPARSE(v)) {
        assert(VEC_PTR_IS_SPARSE(dest));
        assert(v->sparse_nentries <= dest->sparse_cap);
        dest->sparse_nentries = v->sparse_nentries;
        memcpy(dest->data.sparse, v->data.sparse,
               v->sparse_nentries*sizeof(SparseEntry));
    } else {
        assert(VEC_PTR_IS_DENSE(dest));
        memcpy(dest->data.dense, v->data.dense, v->dim*sizeof(real));
    }
}

// macros wrapped in {} instead of do while
// so use macros without ; especially in short if else

#define ZERO_DV(v) memset((v).data.sparse,0,(v).dim*sizeof(real));
#define ZERO_SV(v) (v).sparse_nentries = 0;
void zero_vec(Vec *v) {
    if (VEC_PTR_IS_DENSE(v)) ZERO_DV(*v)
    else ZERO_SV(*v)
}

#if HAS_CBLAS
    #define SCALE_DV(v,a) cblas_scal((v).dim, (a), (v).data.dense, 1);
#else
    #define SCALE_DV(v,a) { \
        if ((a) == 0.0f) ZERO_DV(v)               \
        else                                      \
            for (uint _i = 0; _i < (v).dim; _i++) \
                (v).data.dense[_i] *= (a);        \
    }
#endif

#define SCALE_SV(v,a) { \
    if ((a) == 0.0f) ZERO_SV(v)                                \
    else                                                       \
        for (ushort _ie = 0; _ie < (v).sparse_nentries; _ie++) \
            (v).data.sparse[_ie].val *= (a);                   \
}
void scale_vec(Vec *v, real a) {
    if (VEC_PTR_IS_DENSE(v)) { SCALE_DV(*v, a); }
    else { SCALE_SV(*v, a); }
}

#define SET_DV(v,a) { \
    for (uint _i = 0; _i < (v).dim; _i++) \
        (v).data.dense[_i] = (a);         \
}
#define SET_ENTRIES_SV(v,a) { \
    if ((a) == 0.0f) ZERO_SV(v)                                \
    else                                                       \
        for (ushort _ie = 0; _ie < (v).sparse_nentries; _ie++) \
            (v).data.sparse[_ie].val = (a);                    \
}

#if HAS_CBLAS
    #define ADD_SCALED_DV_TO_DV(u,a,v) { \
        assert((v).dim == (u).dim); \
        cblas_axpy((v).dim, (a), (u).data.dense, 1, (v).data.dense, 1); \
    }
#else
    #define ADD_SCALED_DV_TO_DV(u,a,v) { \
        assert((v).dim == (u).dim);                             \
        if ((a) != 0.0f)                                        \
            for (uint _i = 0; _i < (v).dim; _i++)               \
                (v).data.dense[_i] += (u).data.dense[_i] * (a); \
    }
#endif

#define ADD_SCALED_SV_TO_DV(u,a,v) { \
    assert((v).dim == (u).dim);                                  \
    if ((a) != 0.0f)                                             \
        for (ushort _ie = 0; _ie < (u).sparse_nentries; _ie++) { \
            SparseEntry _ue = (u).data.sparse[_ie];              \
            (v).data.dense[_ue.idx] += _ue.val * (a);            \
        }                                                        \
}
void scaled_vec_addto_dvec(Vec *u, real a, Vec *v) {
    if (VEC_PTR_IS_DENSE(u)) { ADD_SCALED_DV_TO_DV(*u, a, *v); }
    else { ADD_SCALED_SV_TO_DV(*u, a, *v); }
}

#if HAS_CBLAS
    #define ADD_DV_TO_DV(u,v) ADD_SCALED_DV_TO_DV((u),1.f,(v))
#else
    #define ADD_DV_TO_DV(u,v) { \
        assert((v).dim == (u).dim);                   \
        for (uint _i = 0; _i < (v).dim; _i++)         \
            (v).data.dense[_i] += (u).data.dense[_i]; \
    }
#endif

#define ADD_SV_TO_DV(u,v) { \
    assert((v).dim == (u).dim);                              \
    for (ushort _ie = 0; _ie < (u).sparse_nentries; _ie++) { \
        SparseEntry _ue = (u).data.sparse[_ie];              \
        (v).data.dense[_ue.idx] += _ue.val;                  \
    }                                                        \
}
void vec_addto_dvec(Vec *u, Vec *v) {
    if (VEC_PTR_IS_DENSE(u)) { ADD_DV_TO_DV(*u, *v); }
    else { ADD_SV_TO_DV(*u, *v); }
}

void make_mat(Mat *A, uint nrows, uint ncols, ushort sparse_cap) {
    if ((!nrows) | (!ncols)) {
        A->data.dense = 0;
        return;
    }
    A->nrows = nrows;
    A->ncols = ncols;
    A->sparse_nentries = 0;
    A->sparse_cap = sparse_cap;
    if (!sparse_cap) {
        A->data.dense = custom_malloc(nrows * ncols * sizeof(real));
    } else {
        assert(sparse_cap <= nrows * ncols);
        A->data.sparse = custom_malloc(sparse_cap * sizeof(SparseEntry));
    }
}

extern inline void free_mat(Mat *A) {free(A->data.dense);}

void push_entry_smat(Mat *A, uint row, uint col, real val) {
    assert(A->sparse_cap >= A->sparse_nentries + 1);
    assert(row < A->nrows && col < A->ncols);
    ushort ie = A->sparse_nentries++;
    A->data.sparse[ie].idx = row*A->ncols + col;
    A->data.sparse[ie].val = val;
}


#define MAT_IS_SPARSE(m) ((m).sparse_cap != 0)
#define MAT_IS_DENSE(m) (!(m).sparse_cap)
#define MAT_PTR_IS_SPARSE(m) ((m)->sparse_cap != 0)
#define MAT_PTR_IS_DENSE(m) (!(m)->sparse_cap)

#define MAT_IS_SQUARE(m) ((m).nrows == (m).ncols)

#define FLOAT_EPSILON 1e-7f

int dmat_is_symm(Mat *A) {
    if (!MAT_IS_SQUARE(*A))
        return 0;
    uint m = A->nrows, n = A->ncols;
    for (uint i = 0; i < m-1; i++)
        for (uint j = i+1; j < n; j++)
            if (!req(A->data.dense[i*n+j],
                     A->data.dense[j*n+i], FLOAT_EPSILON)) {
                printf("asymm at i=%u j=%u\n", i, j);
                return 0;
            }
    return 1;
}

#define ZERO_DM(m) memset((m).data.sparse,0,(m).nrows*(m).ncols*sizeof(real));
#define ZERO_SM(m) (m).sparse_nentries = 0;
void zero_mat(Mat *A) {
    if (MAT_PTR_IS_DENSE(A)) ZERO_DM(*A)
    else ZERO_SM(*A)
}

void set_dmat_diag(Mat *A, real a) {
    uint n = rmin(A->nrows, A->ncols);
    for (uint i = 0; i < n; i++)
        A->data.dense[i*n+i] = a;
}

void diag_mat_dmat(Mat *A, real a) {
    uint n = A->nrows;
    assert(n == A->ncols);
    for (uint i = 0; i < n; i++)
        for (uint j = 0; j < n; j++)
            A->data.dense[i*n+j] = (i == j)*a;
}

// uTv
real inner_vec(Vec *u, Vec *v) {
    assert(u->dim == v->dim);
    real ret = 0.0f;
    Vec *tmp;
    uint vie, uie;
    switch (VEC_PTR_IS_SPARSE(v)*2 + VEC_PTR_IS_SPARSE(u)) {
    case 0: // both dense
        #if HAS_CBLAS
            ret = cblas_dot(u->dim, u->data.dense, 1, v->data.dense, 1);
        #else
            for (uint i = 0; i < v->dim; i++)
                ret += u->data.dense[i] * v->data.dense[i];
        #endif
        break;
    case 1: // u sparse v dense
        tmp = v; // just swap, fall through to 2
        v = u;
        u = tmp;
    case 2: // u dense v sparse
        for (ushort ie = 0; ie < v->sparse_nentries; ie++) {
            SparseEntry ve = v->data.sparse[ie];
            ret += u->data.dense[ve.idx] * ve.val;
        }
        break;
    case 3: // both sparse
        vie = 0;
        uie = 0;
        // assumes sparse vecs sorted by idx for O(n)
        while ((vie < v->sparse_nentries) && (uie < u->sparse_nentries)) {
            SparseEntry ve = v->data.sparse[vie], ue = u->data.sparse[uie];
            if (ve.idx == ue.idx)
                ret += ve.val * ue.val;
            if (ve.idx < ue.idx)
                vie++;
            else
                uie++;
        }
    }
    return ret;
}

void scaled_self_outer_vec_addto_dmat(Vec *v, real a, Mat *dest) {
    uint n = v->dim;
    assert(n == dest->nrows && n == dest->ncols);
    assert(MAT_PTR_IS_DENSE(dest));
    if (VEC_PTR_IS_DENSE(v)) {
        // if using cblas_?syr then it's just exclusively writing results
        // to only upper or lower matrix, will need to copy, else use
        // cblas_?ger instead
        #if HAS_CBLAS
            cblas_ger(CblasRowMajor, n, n, a,
                      v->data.dense, 1, v->data.dense, 1,
                      dest->data.dense, n);
        #else
            for (uint i = 0; i < n; i++)
                for (uint j = 0; j < n; j++)
                    dest->data.dense[i*n+j] += a * v->data.dense[i]
                                                 * v->data.dense[j];
        #endif
    } else {
        for (uint ie = 0; ie < v->sparse_nentries; ie++) {
            SparseEntry vie = v->data.sparse[ie];
            for (uint je = 0; je < v->sparse_nentries; je++) {
                SparseEntry vje = v->data.sparse[je];
                dest->data.dense[vie.idx*n+vje.idx] += a * vie.val * vje.val;
            }
        }
    }
}

// this and a few other vector math routines couldve benefitted from some
// iterators. that however wasnt convincing enough for me to write them.
void outer_vec_writo_mat(Vec *u, Vec *v, Mat *dest) {
    uint n = u->dim, m = v->dim;
    assert(n == dest->nrows && m == dest->ncols);
    if (MAT_PTR_IS_DENSE(dest)) {
        switch (VEC_PTR_IS_SPARSE(v)*2 + VEC_PTR_IS_SPARSE(u)) {
        case 0: // both dense
                #if HAS_CBLAS
                    ZERO_DM(*dest)
                    cblas_ger(CblasRowMajor, n, m, 1.f, u->data.dense, 1,
                              v->data.dense, 1, dest->data.dense, m);
                #else
                    for (uint i = 0; i < n; i++)
                        for (uint j = 0; j < m; j++)
                            dest->data.dense[i*m+j] = u->data.dense[i]
                                                    * v->data.dense[j];
                #endif
            return;
        case 1: // u sparse v dense
            ZERO_DM(*dest)
            for (ushort ie = 0; ie < u->sparse_nentries; ie++) {
                SparseEntry uie = u->data.sparse[ie];
                for (uint j = 0; j < m; j++)
                    dest->data.dense[uie.idx*m+j] = uie.val
                                                  * v->data.dense[j];
            }
            return;
        case 2: // u dense v sparse
            ZERO_DM(*dest)
            for (uint i = 0; i < n; i++)
                for (ushort je = 0; je < v->sparse_nentries; je++) {
                    SparseEntry vje = v->data.sparse[je];
                    dest->data.dense[i*m+vje.idx] = u->data.dense[i]
                                                  * vje.val;
                }
            return;
        case 3: // both sparse
            ZERO_DM(*dest);
            for (ushort ie = 0; ie < u->sparse_nentries; ie++) {
                SparseEntry uie = u->data.sparse[ie];
                for (ushort je = 0; je < v->sparse_nentries; je++) {
                    SparseEntry vje = v->data.sparse[je];
                    dest->data.dense[uie.idx*m+vje.idx] = uie.val * vje.val;
                }
            }
            return;
        }
    }
    // sparse dest, the thing that makes this function long
    ZERO_SM(*dest);
    switch (VEC_PTR_IS_SPARSE(v)*2 + VEC_PTR_IS_SPARSE(u)) {
    case 0: // both dense
        fputs("really shouldn't be storing dvec dvec^T in smat", stderr);
        assert(dest->sparse_cap >= n*m);
        for (uint i = 0; i < n; i++)
            for (uint j = 0; j < m; j++)
                push_entry_smat(dest, i, j,
                               u->data.dense[i] * v->data.dense[j]);
        return;
    case 1: // u sparse v dense
        assert(dest->sparse_cap >= u->sparse_nentries*m);
        for (ushort ie = 0; ie < u->sparse_nentries; ie++) {
            SparseEntry uie = u->data.sparse[ie];
            for (uint j = 0; j < m; j++)
                push_entry_smat(dest, uie.idx, j, uie.val * v->data.dense[j]);
        }
        return;
    case 2: // u dense v sparse
        assert(dest->sparse_cap >= u->sparse_nentries*m);
        for (uint i = 0; i < n; i++)
            for (ushort je = 0; je < v->sparse_nentries; je++) {
                SparseEntry vje = v->data.sparse[je];
                push_entry_smat(dest, i, vje.idx, u->data.dense[i] * vje.val);
            }
        return;
    case 3: // both sparse
        assert(dest->sparse_cap >= u->sparse_nentries*v->sparse_nentries);
        for (ushort ie = 0; ie < u->sparse_nentries; ie++) {
            SparseEntry uie = u->data.sparse[ie];
            for (ushort je = 0; je < v->sparse_nentries; je++) {
                SparseEntry vje = v->data.sparse[je];
                push_entry_smat(dest, uie.idx, vje.idx, uie.val * vje.val);
            }
        }
        return;
    }
}

extern inline
void flat_outer_vec_writo_vec(Vec *u, Vec *v, Vec *dest) {
    Mat m;
    assert(dest->dim == u->dim*v->dim);
    m.nrows = u->dim;
    m.ncols = v->dim;
    m.data.dense = dest->data.dense;
    m.sparse_cap = dest->sparse_cap;
    outer_vec_writo_mat(u, v, &m);
    dest->sparse_nentries = m.sparse_nentries;
    dest->sparse_cap = m.sparse_cap; // just in case in the future
}


void mat_addto_dmat(Mat *A, Mat *dest) {
    uint n = A->nrows, m = A->ncols;
    assert(dest->nrows == n && dest->ncols == m);
    assert(MAT_PTR_IS_DENSE(dest));
    if (MAT_PTR_IS_DENSE(A)) {
        for (uint i = 0; i < n; i++)
            for (uint j = 0; j < m; j++)
                dest->data.dense[i*m+j] += A->data.dense[i*m+j];
    } else {
        for (ushort ie = 0; ie < A->sparse_nentries; ie++) {
            SparseEntry Aie = A->data.sparse[ie];
            dest->data.dense[Aie.idx] += Aie.val;
        }
    }
}

// A_{m x n} v_{n x 1}
void dmat_mul_vec_writo_dvec(Mat *A, Vec *v, Vec *dest) {
    uint n = A->nrows, m = A->ncols;
    assert(n == dest->dim && m == v->dim);
    assert(MAT_PTR_IS_DENSE(A) && VEC_PTR_IS_DENSE(dest));
    assert(v != dest);

    if (VEC_PTR_IS_DENSE(v)) {
        #if HAS_CBLAS
            // y = beta*y + alpha*A*x
            cblas_gemv(CblasRowMajor, CblasNoTrans, n, m,
                       1.f, // alpha
                       A->data.dense, m, // A
                       v->data.dense, 1, // x
                       0.f, dest->data.dense, 1); // beta, y
        #else
            for (uint i = 0; i < n; i++) {
                dest->data.dense[i] = 0.0f;
                for (uint j = 0; j < m; j++)
                    dest->data.dense[i] += A->data.dense[i*m+j]
                                         * v->data.dense[j];
            }
        #endif
    } else {
        for (uint i = 0; i < m; i++) {
            dest->data.dense[i] = 0.0f;
            for (uint je = 0; je < v->sparse_nentries; je++) {
                SparseEntry ve = v->data.sparse[je];
                dest->data.dense[i] += A->data.dense[i*m + ve.idx] * ve.val;
            }
        }
    }
}

void print_vec(Vec *v) {
    printf("%cvec: dim=%u, spn=%u/%u\t",
           v->sparse_cap ? 's' : 'd',
           v->dim, v->sparse_nentries, v->sparse_cap);
    if (VEC_PTR_IS_DENSE(v)) {
        printf("(");
        for (uint i = 0; i < v->dim; i++)
            printf("%.3g%c ", v->data.dense[i], i == v->dim-1 ? ')' : ',');
        printf("\n");
    } else {
        for (ushort ie = 0; ie < v->sparse_nentries; ie++) {
            SparseEntry ve = v->data.sparse[ie];
            printf("[%u]=%.4g  ", ve.idx, ve.val);
        }
        printf("\n");
    }
}

void print_mat(Mat *A) {
    printf("%cmat: dim=%ux%u, spn=%u/%u ",
           A->sparse_cap ? 's' : 'd',
           A->nrows, A->ncols, A->sparse_nentries, A->sparse_cap);
    if (MAT_PTR_IS_DENSE(A)) {
        putchar('\n');
        for (uint i = 0; i < A->nrows; i++) {
            for (uint j = 0; j < A->ncols; j++)
                printf("%5.3f ", A->data.dense[i*A->ncols+j]);
            printf("\n");
        }
    } else {
        putchar('\t');
        for (ushort ie = 0; ie < A->sparse_nentries; ie++) {
            SparseEntry Aie = A->data.sparse[ie];
            printf("[%u,%u]=%.4g  ",
                   Aie.idx/A->ncols, Aie.idx%A->ncols, Aie.val);
        }
        printf("\n");
    }
}
