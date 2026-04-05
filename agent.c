// -------------------- Agent ----------------------------------

typedef struct {
    void (*choose_action)(void *self_, RngState *rngs,
                          Elem *S, Elem *A, uint t);
    void (*update)       (void *self_, RngState *rngs,
                          Elem *S, Elem *A, real R, Elem *nS,
                          uint t, bool is_terminal);
    void (*start_ep)     (void *self_, RngState *rngs, Elem *S_0);
    // might not be called for continuing mdp
    void (*end_ep)       (void *self_, RngState *rngs, uint T);
    void (*reset)        (void *self_);
    // why not
    char *name;
} AgentVT;

typedef struct {
    AgentVT VT_PTR_ASTER vt;
    Environment *env;
} Agent;

// -------------------- SarsaLambda ---------------------------

typedef struct {
    Agent super;
    QFn *qfn;
    Policy *pi;
    Vec elig;
    Elem nA;
    float alpha, gamma, lambda;
} SarsaLambda;

void SarsaLambda_start_ep(void *self_, RngState *rngs, Elem *S_0) {
    SarsaLambda *self = self_;
    zero_vec(&self->elig);
    self->pi->VT_ACCESS choose_action(self->pi, rngs, S_0, &self->nA);
}

void SarsaLambda_end_ep(void *self_, RngState *rngs, uint T) {}

void SarsaLambda_choose_action(void *self_, RngState *rngs, Elem *S, Elem *A,
                               uint t) {
    SarsaLambda *self = self_;
    copy_elem_to_elem(&self->nA,
                      &self->super.env->SVT_ACCESS fixed_action_space, A);
}
uint qg = 0;
void SarsaLambda_update(void *self_, RngState *rngs,
                        Elem *S, Elem *A, real R, Elem *nS,
                        uint t, bool is_terminal) {
    SarsaLambda *self = self_;
    float QSA = Qtheta(self->qfn, S, A);
    float delta = R - QSA;
    if (!is_terminal) {
        self->pi->VT_ACCESS choose_action(self->pi, rngs, nS, &self->nA);
        delta += self->gamma * Qtheta(self->qfn, nS, &self->nA);
    }
    scale_vec(&self->elig, self->gamma * self->lambda);
    dQtheta_dtheta_addto_dvec(self->qfn, S, A, &self->elig);
    scaled_vec_addto_dvec(&self->elig, self->alpha * delta, &self->qfn->theta);
    // float vmul = self->alpha*delta;
    // qg++;
    // if (qg % 100 == 0) {
    //     float normv = inner_vec(&self->elig, &self->elig);
    //     float normelig = inner_vec(&self->elig, &self->elig);
    //     float bilin = inner_vec(&self->elig, &self->elig);
    //     printf("delta=%.6f\t"
    //         "change=%.6f\t"
    //         "L2(update)=%.6f\t"
    //         "cos(ang(v,elig))=%.6f\n",
    //         delta, QSA - Qtheta(self->qfn, S, A),
    //         normv*vmul, 180/M_PI*acosf(bilin/sqrtf(normv*normelig)));
    // }
}

void SarsaLambda_reset(void *self_) {
    SarsaLambda *self = self_;
    // internal states
    ZERO_DV(self->elig)
    // qfn
    ZERO_DV(self->qfn->theta)
    // reset policy: we'll get to it when we get to it
}

AgentVT SarsaLambda_vt = {
    .name = "SarsaLambda",
    .start_ep = SarsaLambda_start_ep,
    .choose_action = SarsaLambda_choose_action,
    .update = SarsaLambda_update,
    .end_ep = SarsaLambda_end_ep,
    .reset = SarsaLambda_reset
};


void make_SarsaLambda(SarsaLambda *self, Environment *env, QFn *qfn,
                      Policy *pi, float alpha, float gamma, float lambda) {
    self->super.vt = VT_PTR_AMPER SarsaLambda_vt;
    self->super.env = env;
    self->qfn = qfn;
    self->pi = pi;
    self->alpha = alpha;
    self->gamma = gamma;
    self->lambda = lambda;
    make_vec(&self->elig, qfn->approx_arch->indim, 0);
    assert(env->SVT_ACCESS action_space_is_fixed);
    make_elem(&self->nA, &env->SVT_ACCESS fixed_action_space);
}

void free_SarsaLambda(SarsaLambda *self) {
    free_vec(&self->elig);
    free_elem(&self->nA);
}

// -------------------- QTNatSarsaLambda ---------------------------

typedef struct {
    Agent super;
    QFn *qfn;
    Policy *pi;
    Mat Ginv;
    Vec elig, g, v;
    Elem nA;
    float alpha, gamma, lambda;
} QTNatSarsaLambda;

void QTNatSarsaLambda_start_ep(void *self_, RngState *rngs, Elem *S_0) {
    QTNatSarsaLambda *self = self_;
    zero_vec(&self->elig);
    self->pi->VT_ACCESS choose_action(self->pi, rngs, S_0, &self->nA);
}

void QTNatSarsaLambda_end_ep(void *self_, RngState *rngs, uint T) {}

void QTNatSarsaLambda_choose_action(void *self_, RngState *rngs,
                                    Elem *S, Elem *A, uint t) {
    QTNatSarsaLambda *self = self_;
    copy_elem_to_elem(&self->nA,
                      &self->super.env->SVT_ACCESS fixed_action_space, A);
}
uint qq = 0;
void QTNatSarsaLambda_update(void *self_, RngState *rngs,
                             Elem *S, Elem *A, real R, Elem *nS,
                             uint t, bool is_terminal) {
    QTNatSarsaLambda *self = self_;
    // calculte delta
    float QSA = Qtheta(self->qfn, S, A);
    float delta = R - QSA;
    if (!is_terminal) {
        self->pi->VT_ACCESS choose_action(self->pi, rngs, nS, &self->nA);
        delta += self->gamma * Qtheta(self->qfn, nS, &self->nA);
    }

    // calculate g
    dQtheta_dtheta_writo_vec(self->qfn, S, A, &self->g);

        // update e
        scale_vec(&self->elig, self->gamma*self->lambda);
        ADD_DV_TO_DV(self->g, self->elig)


    dQtheta_dtheta_writo_vec(self->qfn, nS, &self->nA, &self->v);
    scaled_vec_addto_dvec(&self->v, -self->gamma, &self->g);


    // v = G^{-1} g
    dmat_mul_vec_writo_dvec(&self->Ginv, &self->g, &self->v);

    // update G^{-1} with Sherman-Morrison inverse
    float scale = -delta*delta
                / (1 + delta*delta*inner_vec(&self->g, &self->v));
    scaled_self_outer_vec_addto_dmat(&self->v, scale, &self->Ginv);
    qq++;
    // update theta with natural semigradient
    dmat_mul_vec_writo_dvec(&self->Ginv, &self->elig, &self->v);
    float vmul = self->alpha*delta;
    scaled_vec_addto_dvec(&self->v, vmul, &self->qfn->theta);
    // yep its always symmetric
    // if (!dmat_is_symm(&self->Ginv)) {
    //     print_mat(&self->Ginv);
    //     exit(1);
    // }
    if (0) {
        float normv = sqrtf(inner_vec(&self->v, &self->v));
        float normelig = sqrtf(inner_vec(&self->elig, &self->elig));
        float bilin = inner_vec(&self->v, &self->elig);
        print_mat(&self->Ginv);
        printf(
            // "delta=%.6f\t"
            // "change=%.6f\t"
            "L2(v)=%.6f\t"
            "|v|/|elig|=%.6f\t"
            "ang(v,elig)=%.6f\n",
            // delta,
            // QSA - Qtheta(self->qfn, S, A),
            normv*vmul,
            normv / normelig,
            180/M_PI*acosf(bilin/(normv*normelig)));
    }
}

void reset_QTNatSarsaLambda(void *self_) {
    QTNatSarsaLambda *self = self_;
    // internal states
    // g and v are moreso scratch vecs for calculations, no need to reset
    // since they do not meaningfully carry across from timesteps anyway
    ZERO_DV(self->elig)
    diag_mat_dmat(&self->Ginv, 1);
    // qfn
    ZERO_DV(self->qfn->theta)
    // reset policy: we'll get to it when we get to it
}

AgentVT QTNatSarsaLambda_vt = {
    .name = "QTNatSarsaLambda",
    .start_ep = QTNatSarsaLambda_start_ep,
    .choose_action = QTNatSarsaLambda_choose_action,
    .update = QTNatSarsaLambda_update,
    .end_ep = QTNatSarsaLambda_end_ep,
    .reset = reset_QTNatSarsaLambda
};

void make_QTNatSarsaLambda(QTNatSarsaLambda *self, Environment *env, QFn *qfn,
                      Policy *pi, float alpha, float gamma, float lambda) {
    self->super.vt = VT_PTR_AMPER QTNatSarsaLambda_vt;
    self->super.env = env;
    self->qfn = qfn;
    self->pi = pi;
    self->alpha = alpha;
    self->gamma = gamma;
    self->lambda = lambda;

    uint d = qfn->approx_arch->indim;
    make_vec(&self->elig, d, 0);
    make_vec(&self->g, d, 0);
    make_vec(&self->v, d, 0);
    make_mat(&self->Ginv, d, d, 0);
    diag_mat_dmat(&self->Ginv, 1);
    assert(env->SVT_ACCESS action_space_is_fixed);
    make_elem(&self->nA, &env->SVT_ACCESS fixed_action_space);
}

void free_QTNatSarsaLambda(QTNatSarsaLambda *self) {
    free_vec(&self->elig);
    free_vec(&self->g);
    free_vec(&self->v);
    free_mat(&self->Ginv);
    free_elem(&self->nA);
}
