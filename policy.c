// -------------------- Policy --------------------------------

typedef struct {
    void (*choose_action)(void *self_, RngState *s, Elem *S, Elem *A);
} PolicyVT;

typedef struct {
    PolicyVT VT_PTR_ASTER vt;
    Environment *env;
} Policy;

// -------------------- FirstAction ---------------------------

typedef struct {
    Policy super;
} FirstAction;

void FirstAction_choose_action(void *self_, RngState *s, Elem *S, Elem *A) {
    FirstAction *self = self_;
    assert(self->super.env->SVT_ACCESS action_space_is_fixed);
    uint nfactors = self->super.env->SVT_ACCESS fixed_action_space.nfactors;
    for (uint i = 0; i < nfactors; i++)
        A->x[i].i = 0;
}

PolicyVT FirstAction_vt = {
    .choose_action = FirstAction_choose_action,
};

void make_FirstAction(FirstAction *self, Environment *env) {
    self->super.vt = VT_PTR_AMPER FirstAction_vt;
    self->super.env = env;
}

// -------------------- EpGreedy -------------------------------

typedef struct {
    Policy super;
    QFn *Q;
    Elem nA;
    float epsilon;
} EpGreedy;


void EpGreedy_choose_action(void *self_, RngState *rngs, Elem *S, Elem *A) {
    EpGreedy *self = self_;
    Space aspace = self->super.env->SVT_ACCESS fixed_action_space;
    uint nfactors = aspace.nfactors;
    if (rand_bernoulli(rngs, self->epsilon)) {
        // ignore bias from u64 max, it's negligible
        for (uint i = 0; i < nfactors; i++)
            A->x[i].i = rand_u64(rngs) % aspace.factors[i].as.discrete.n;
        // printf("Epgreedy explore: ");
        // print_elem(A, &self->super.env->SVT_ACCESS fixed_action_space);
        return;
    }

    #if 0
        // greedy, with reverse order scan (first elem of tuple increased first)
        // dont require next max to be bigger than current max by some epsilon
        memset(self->nA.x, 0, nfactors * sizeof(SimpleElem));
        float max_value = FLOAT_VERY_SMALL;

        while (1) {
            float value = Qtheta(self->Q, S, &self->nA);
            if (value > max_value) {
                copy_elem_to_elem(&self->nA,
                                  &self->super.env->SVT_ACCESS fixed_action_space,
                                  A);
                max_value = value;
            }
            self->nA.x[0].i++;
            for (uint i = 0; i < nfactors - 1; i++) {
                if (self->nA.x[i].i >= aspace.factors[i].as.discrete.n) {
                    self->nA.x[i].i = 0;
                    self->nA.x[i+1].i++;
                }
            }
            if (self->nA.x[nfactors-1].i >=
                aspace.factors[nfactors-1].as.discrete.n)
                break;
        }
        // printf("Epgreedy greedy: ");
        // print_elem(A, &self->super.env->SVT_ACCESS fixed_action_space);
    #else
        #define TIE_EPSILON 0.0000001f
        memset(self->nA.x, 0, nfactors * sizeof(SimpleElem));
        float max_value = FLOAT_VERY_SMALL;
        uint nmax_actions = 0;

        while (1) {
            float value = Qtheta(self->Q, S, &self->nA);
            if (value > max_value + TIE_EPSILON) {
                copy_elem_to_elem(&self->nA,
                                  &self->super.env->SVT_ACCESS fixed_action_space,
                                  A);
                max_value = value;
                nmax_actions = 1;
            } else if (max_value - TIE_EPSILON < value) {
                nmax_actions++;
            }
            self->nA.x[0].i++;
            for (uint i = 0; i < nfactors - 1; i++) {
                if (self->nA.x[i].i >= aspace.factors[i].as.discrete.n) {
                    self->nA.x[i].i = 0;
                    self->nA.x[i+1].i++;
                }
            }
            if (self->nA.x[nfactors-1].i >=
                aspace.factors[nfactors-1].as.discrete.n)
                break;
        }

        if (nmax_actions == 1)
            return;

        // randomly pick one of the tied maximizing actions
        uint picked_maxer_order = rand_u64(rngs) % nmax_actions;
        memset(self->nA.x, 0, nfactors * sizeof(SimpleElem));
        while (1) {
            float value = Qtheta(self->Q, S, &self->nA);
            if (max_value - TIE_EPSILON < value
                && value < max_value + TIE_EPSILON)
            {
                picked_maxer_order--;
                if (picked_maxer_order == 0) {
                    copy_elem_to_elem(&self->nA,
                      &self->super.env->SVT_ACCESS fixed_action_space, A);
                    return;
                }
            }
            self->nA.x[0].i++;
            for (uint i = 0; i < nfactors - 1; i++) {
                if (self->nA.x[i].i >= aspace.factors[i].as.discrete.n) {
                    self->nA.x[i].i = 0;
                    self->nA.x[i+1].i++;
                }
            }
            if (self->nA.x[nfactors-1].i >=
                aspace.factors[nfactors-1].as.discrete.n)
                break;
        }
    #endif
}

PolicyVT EpGreedy_vt = {
    .choose_action = EpGreedy_choose_action,
};

void make_EpGreedy(EpGreedy *self, Environment *env, QFn *Q, float epsilon) {
    self->super.vt = VT_PTR_AMPER EpGreedy_vt;
    self->super.env = env;
    self->epsilon = epsilon;
    self->Q = Q;
    make_elem(&self->nA, &env->SVT_ACCESS fixed_action_space);
    assert(env->SVT_ACCESS action_space_is_fixed);
    assert(env->SVT_ACCESS action_space_is_discrete);
    assert(0.f <= epsilon && epsilon <= 1.f);
}

extern inline
void free_EpGreedy(EpGreedy *self) {free_elem(&self->nA);}
