// -------------------- SmoothParametricFn --------------------

// Smooth parametric function (approximation architecture)
// for simple functions this might just look like a glorified vtable, but
// need these as classes that can have extra metadata, e.g. if its a neural
// net then what do the layers look like etc.
// still, these do not own the parameter vec theta. the QFn or sth else does.
// still. these are meant to also be able to work on their own without a QFn.

typedef struct {
    real (*f)                 (void *self_, Vec *theta, Vec *x);
    void (*deriv_writo_vec)   (void *self_, Vec *theta, Vec *x, Vec *dest);
    void (*deriv_addto_dvec)  (void *self_, Vec *theta, Vec *x, Vec *dest);
    void (*prealloc_deriv_vec)(void *self_,                     Vec *dest);
    void (*alloc_init_theta)  (void *self_, Vec *theta);
} SmoothParametricFnVT;

typedef struct {
    SmoothParametricFnVT VT_PTR_ASTER vt;
    uint indim, flags;
} SmoothParametricFn;

// linear approximation

typedef struct {
    SmoothParametricFn super;
} Linear;

void Linear_alloc_init_theta(void *self_, Vec *theta) {
    Linear *self = self_;
    // hmm sparse theta? probably not useful for now...
    make_vec(theta, self->super.indim, 0);
    ZERO_DV(*theta);
}

real Linear_f(void *self_, Vec *theta, Vec *x) {
    return inner_vec(theta, x);
}

void Linear_deriv_wri(void *self, Vec *theta, Vec *x, Vec *dest) {
    copy_vec_to_vec(x, dest);
}

void Linear_deriv_add(void *self, Vec *theta, Vec *x, Vec *dest) {
    vec_addto_dvec(x, dest);
}

void Linear_prealloc_deriv(void *self_, Vec *dest) {
    // does nothing.
    assert(VEC_NOT_ALLOCD(dest));
}

SmoothParametricFnVT Linear_vt = {
    .f = Linear_f,
    .deriv_writo_vec = Linear_deriv_wri,
    .deriv_addto_dvec = Linear_deriv_add,
    .alloc_init_theta = Linear_alloc_init_theta,
    .prealloc_deriv_vec = Linear_prealloc_deriv
};

void make_Linear(Linear *self, uint indim) {
    self->super.vt = VT_PTR_AMPER Linear_vt;
    self->super.indim = indim;
}

extern inline
void free_Linear(Linear *self) {}
