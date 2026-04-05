// -------------------- FeatureMap ----------------------------

// Feature mappings from \mathcal{S} to R^{d_S} or \mathcal{A} to R^{d_A}

typedef struct {
    void (*map_writo_vec)   (void *self_, Elem *x, Vec *dest);
    void (*prealloc_out_vec)(void *self_,          Vec *dest);
} FeatureMapVT;

typedef struct {
    FeatureMapVT VT_PTR_ASTER vt;
    Space *space;
    uint outdim, flags;
} FeatureMap;

// -------------------- OneHot --------------------------------

// one-hot encoder for product of discrete sets

typedef struct {
    FeatureMap super;
} OneHot;

#define ONEHOT_FL_PUSH 1

void OneHot_map(void *self_, Elem *x, Vec *dest) {
    OneHot *self = self_;
    Space space = *self->super.space;
    uint hot_idx = 0;
    for (uint i = 0; i < space.nfactors; i++) {
        hot_idx *= space.factors[i].as.discrete.n;
        hot_idx += x->x[i].i;
    }
    if (self->super.flags & ONEHOT_FL_PUSH) {
        // allow for putting multiple hot entries
        push_entry_svec(dest, hot_idx, 1);
    } else {
        dest->sparse_nentries = 1;
        dest->data.sparse[0].idx = hot_idx;
        dest->data.sparse[0].val = 1;
    }
}

void OneHot_prealloc_out_vec(void *self_, Vec *dest) {
    OneHot *self = self_;
    assert(VEC_NOT_ALLOCD(dest));
    make_vec(dest, self->super.outdim, 1);
}

FeatureMapVT OneHot_vt = {
    .prealloc_out_vec = OneHot_prealloc_out_vec,
    .map_writo_vec = OneHot_map
};

void make_OneHot(OneHot *self, Space *space, uint flags) {
    self->super.vt = VT_PTR_AMPER OneHot_vt;
    self->super.space = space;
    self->super.outdim = get_finite_space_cardinality(space);
    self->super.flags = flags; // just dont actually use this just yet
}

extern inline
void free_OneHot(OneHot *self) {}

// -------------------- FourierBasis --------------------------

// Fourier features as described in "Value Function Approximation in
// Reinforcement Learning using the Fourier Basis" by Konidaris, Osentoski,
// Thomas. that is, only cosine basis functions, scaled to match specified
// ranges. also can specify order for each dimension separately.

typedef struct {
    FeatureMap super;
    // this one needs to store space. realistically, it can also get the
    // space from the Elem that was passed in, but also maybe at some point
    // Elem won't have to contain space.
    uint *orders; // order is not copied or anything. user freed.
    void *helpers;
} FourierBasis;

void FoureirBasis_prealloc_out_vec(void *self_, Vec *dest) {
    FourierBasis *self = self_;
    assert(VEC_NOT_ALLOCD(dest));
    make_vec(dest, self->super.outdim, 0);
}

void FourierBasis_map(void *self_, Elem *x, Vec *dest) {
    FourierBasis *self = self_;
    assert(VEC_PTR_IS_DENSE(dest));

    Space space = *self->super.space;
    uint d = space.nfactors;
    // backward-cumulative sum of dot product terms
    // e.g. cum_sum[0] + x[0]*c[0] = x dot product c
    real *cum_sum = self->helpers;
    uint *cur_coeffs = (void*)((d)*sizeof(real) + (char*)cum_sum);
    real *xnormal = (void*)(d*sizeof(uint) + (char*)cur_coeffs);
    memset(self->helpers, 0,
        (d)*sizeof(real) + (d)*sizeof(uint));
    // really only need it to have a range of length 1,
    // not specifically [0, 1]
    for (uint i = 0; i < d; i++) {
        Range range = space.factors[i].as.range;
        xnormal[i] = (x->x[i].r - range.a)
                   / (range.b - range.a);
    }

    uint last_changed = 0;
    for (uint iter = 0;;) {

        real xdotc = xnormal[last_changed]*cur_coeffs[last_changed]
                         + cum_sum[last_changed];

        dest->data.dense[iter] = cosf(M_PI * xdotc);

        iter++;
        if (iter == self->super.outdim)
            break;

        for (uint i = 0; i < last_changed; i++)
            cum_sum[i] = xdotc;

        cur_coeffs[0]++;
        last_changed = 0;
        while (last_changed < d-1
            && (cur_coeffs[last_changed] > self->orders[last_changed]))
        {
            cur_coeffs[last_changed] = 0;
            cur_coeffs[last_changed+1]++;
            last_changed++;
        }
    }
}

FeatureMapVT FourierBasis_vt = {
    .prealloc_out_vec = FoureirBasis_prealloc_out_vec,
    .map_writo_vec = FourierBasis_map
};

void make_FourierBasis(FourierBasis *self, Space *space, uint *orders) {
    self->super.vt = VT_PTR_AMPER FourierBasis_vt;
    self->super.space = space;
    self->orders = orders;
    self->super.outdim = 1;
    for (uint i = 0; i < space->nfactors; i++) {
        assert(space->factors[i].type == 'r');
        self->super.outdim *= orders[i] + 1;
    }
    // kinda like arena allocation
    self->helpers = custom_malloc(
        // backward-cumulative sum and range-normalized element
        2*(space->nfactors) * sizeof(real)
        + (space->nfactors) * sizeof(uint) // current coefficients
    );
}

void free_FourierBasis(FourierBasis *self) {free(self->helpers);}

// -------------------- SAFeatureMap --------------------------

// Feature mappings from technically from \mathcal{S}\times\mathcal{A} to
// R^{d_S d_A} but S and A passed separately.
// combining separate feature maps for S and A with outer product is the only
// one we're using for now.

typedef struct {
    void (*map_writo_vec)   (void *self_, Elem *S, Elem *A, Vec *dest);
    void (*prealloc_out_vec)(void *self_,                   Vec *dest);
} SAFeatureMapVT;

typedef struct {
    SAFeatureMapVT VT_PTR_ASTER vt;
    uint outdim, flags;
} SAFeatureMap;

// -------------------- CombineSAFeatureMaps ------------------

// Combine S and A feature maps into (S,A) feature map
// this one is not really meant to be used directly
// todo at some point: just concat the featuremaps (map to R^{d_S + d_A})

typedef struct {
    SAFeatureMap super;
    void (*combine_writo_vec)(void *self_, Vec *dest);
    FeatureMap *s_feamap, *a_feamap;
    Vec fea_S, fea_A;
} CombineSAFeatureMaps;


static
void CombineSAFeatureMaps_map(void *self_, Elem *S, Elem *A, Vec *dest) {
    CombineSAFeatureMaps *self = self_;
    self->s_feamap->VT_ACCESS map_writo_vec(self->s_feamap, S, &self->fea_S);
    self->a_feamap->VT_ACCESS map_writo_vec(self->a_feamap, A, &self->fea_A);
    self->combine_writo_vec(self, dest);
}

void _make_CombineSAFeatureMaps(void *self_, void *s_feamap, void *a_feamap,
                                void (*combine)(void*,Vec*), uint outdim,
                                uint flags) {
    CombineSAFeatureMaps *self = self_;
    self->combine_writo_vec = combine;
    self->s_feamap = s_feamap;
    self->a_feamap = a_feamap;
    make_vec(&self->fea_S, 0, 0);
    make_vec(&self->fea_A, 0, 0);
    self->s_feamap->VT_ACCESS prealloc_out_vec(s_feamap, &self->fea_S);
    self->a_feamap->VT_ACCESS prealloc_out_vec(a_feamap, &self->fea_A);
    self->super.outdim = outdim;
    self->super.flags = flags;
}

void _free_CombineSAFeatureMaps(void *self_) {
    CombineSAFeatureMaps *self = self_;
    free_vec(&self->fea_S);
    free_vec(&self->fea_A);
}

// -------------------- FlatOuterProduct ----------------------

typedef struct {
    CombineSAFeatureMaps super;
} FlatOuterProduct;

#define VEC_EFFECTIVE_DIM(v) \
    ((v).sparse_cap ? (v).sparse_nentries : (v).dim)

#define THRESHOLD_SPARSITY 8
#define SPARSE_ALLOC_FACTOR_WRT_EFFECTIVE_DIM 4
#define MAX_SPARSE_ALLOCATION 1024

#define FLATOUTERPROD_FL_FLIPORDER 1

#define FLATOUTERPROD_PREALLOC_SENTINEL 1234

void FlatOuterProduct_prealloc_out_vec(void *self_, Vec *dest) {
    // this does nothing. the combine function does allocation (and resizing)
    // of the output vector if needed. this just puts a sentinel value to make
    // sure prealloc was called beforehand, instead of checking for the 0 in
    // vec's data as usual.

    dest->data.dense = (void*)FLATOUTERPROD_PREALLOC_SENTINEL;
}

void FlatOuterProduct_combine(void *self_, Vec *dest) {
    FlatOuterProduct *self = self_;

    assert(dest->data.dense);
    if (dest->data.dense == (void*)FLATOUTERPROD_PREALLOC_SENTINEL) {
    // if (VEC_NOT_ALLOCD(dest)) { // the allocor behavior
        uint total_dim = self->super.fea_S.dim * self->super.fea_A.dim;
        // note: this allocor for fea_SA bases effective dimensions calc on
        // the first time it sees fea_S and fea_A, while automatic conversion
        // to/from svec/dvec and reallocation if needed is not implemented yet
        uint effective_dim = VEC_EFFECTIVE_DIM(self->super.fea_S)
                           * VEC_EFFECTIVE_DIM(self->super.fea_A);
        if (effective_dim &&
            total_dim / effective_dim <= THRESHOLD_SPARSITY) {
            make_vec(dest, total_dim, 0);
        } else {
            // reserving space for as many entries as dense is ok, since it's
            // not gonna be much memory anyway (probably)
            // edit: lol that was very wrong. 800M-d unit test.
            make_vec(dest, total_dim, umin(umin(
                SPARSE_ALLOC_FACTOR_WRT_EFFECTIVE_DIM * effective_dim,
                total_dim), MAX_SPARSE_ALLOCATION));
        }
    }
    if (!(self->super.super.flags & FLATOUTERPROD_FL_FLIPORDER))
        flat_outer_vec_writo_vec(&self->super.fea_S,
                                 &self->super.fea_A, dest);
    else
        flat_outer_vec_writo_vec(&self->super.fea_A,
                                 &self->super.fea_S, dest);
}

SAFeatureMapVT FlatOuterProduct_vt = {
    .prealloc_out_vec = FlatOuterProduct_prealloc_out_vec,
    .map_writo_vec = CombineSAFeatureMaps_map
};

void make_FlatOuterProduct(FlatOuterProduct *self, void *s_feamap,
                           void *a_feamap, uint order) {
    _make_CombineSAFeatureMaps(self, s_feamap, a_feamap,
        FlatOuterProduct_combine,
        ((FeatureMap*)s_feamap)->outdim * ((FeatureMap*)a_feamap)->outdim,
        order
    );
    self->super.super.vt = VT_PTR_AMPER FlatOuterProduct_vt;
}

void free_FlatOuterProduct(FlatOuterProduct *self) {
    _free_CombineSAFeatureMaps(&self->super);
}
