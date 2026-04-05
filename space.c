// -------------------- Space and Elem ------------------------


typedef struct {
    uint n;
} Discrete; // { 0...n-1 }
typedef struct {
    real a, b;
} Range; // [a, b]

typedef struct {
    union {
        Discrete discrete;
        Range range;
    } as;
    uchar type; // d for discrete, r for range
} SimpleSet;

typedef union {
    uint i;
    real r;
} SimpleElem;

// cartesian product of simple sets
typedef struct {
    SimpleSet *factors;
    uint nfactors;
} Space;

typedef struct {
    SimpleElem *x;
    // Space *space;
} Elem;

extern inline
void make_space(Space *space, uint nfactors) {
    space->nfactors = nfactors;
    space->factors = malloc(nfactors * sizeof(SimpleSet));
}

extern inline
void free_space(Space *space) {free(space->factors);}

extern inline
void set_space_discrete_factor(Space *space, uint idx, Discrete d) {
    assert(idx < space->nfactors);
    assert(d.n);
    space->factors[idx].as.discrete = d;
    space->factors[idx].type = 'd';
}

extern inline
void set_space_range_factor(Space *space, uint idx, Range r) {
    assert(idx < space->nfactors);
    assert(r.a != r.b);
    space->factors[idx].as.range = r;
    space->factors[idx].type = 'r';
}

extern inline
void make_elem(Elem *x, Space *space) {
    // x->space = space;
    x->x = malloc(space->nfactors * sizeof(SimpleElem));
}

extern inline
void copy_elem_to_elem(Elem *x, Space *space, Elem *dest) {
    memcpy(dest->x, x->x, space->nfactors * sizeof(SimpleElem));
}

extern inline
void free_elem(Elem *x) {free(x->x);}

uint get_finite_space_cardinality(Space *space) {
    uint card = 1;
    for (uint i = 0; i < space->nfactors; i++) {
        assert(space->factors[i].type == 'd');
        card *= space->factors[i].as.discrete.n;
    }
    return card;
}

void print_simple_set(SimpleSet *ss, char *endline) {
    if (ss->type == 'd') {
        printf("i[%u]%s", ss->as.discrete.n, endline);
    } else if (ss->type == 'r') {
        printf("f[%.4g, %.4g]%s", ss->as.range.a, ss->as.range.b, endline);
    } else {
        assert(!"unknown set type");
    }
}

void print_space(Space *space) {
    printf("Space ");
    for (uint i = 0; i < space->nfactors; i++) {
        print_simple_set(&space->factors[i], "");
        if (i != space->nfactors - 1) {
            printf(" x ");
        }
    }
    putchar('\n');
}

void print_elem(Elem *x, Space *space) {
    printf("Elem (");
    for (uint i = 0; i < space->nfactors; i++) {
        uchar type = space->factors[i].type;
        if (type == 'r') {
            printf("%.4gf", x->x[i].r);
        } else if (type == 'd') {
            printf("%ui", x->x[i].i);
        }
        if (i != space->nfactors - 1) {
            printf(", ");
        }
    }
    printf(")\n");

}
