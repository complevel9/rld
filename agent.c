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

// -------------------- ResidualGrad --------------------------

typedef struct {
    Agent super;
    QFn *qfn;
    Policy *pi;
    Vec g;
    Elem nA;
    real alpha, gamma;
} ResidualGrad;

void ResidualGrad_start_ep(void *self_, RngState *rngs, Elem *S_0) {
    ResidualGrad *self = self_;
    self->pi->VT_ACCESS choose_action(self->pi, rngs, S_0, &self->nA);
}

void ResidualGrad_end_ep(void *self_, RngState *rngs, uint T) {}

void ResidualGrad_choose_action(void *self_, RngState *rngs, Elem *S, Elem *A,
                               uint t) {
    ResidualGrad *self = self_;
    copy_elem_to_elem(&self->nA,
                      &self->super.env->SVT_ACCESS fixed_action_space, A);
}

void ResidualGrad_update(void *self_, RngState *rngs,
                         Elem *S, Elem *A, real R, Elem *nS,
                         uint t, bool is_terminal) {
    ResidualGrad *self = self_;
    real delta = R - Qtheta(self->qfn, S, A);
    dQtheta_dtheta_writo_vec(self->qfn, S, A, &self->g);
    if (!is_terminal) {
        self->pi->VT_ACCESS choose_action(self->pi, rngs, nS, &self->nA);
        delta += self->gamma * Qtheta(self->qfn, nS, &self->nA);
        scaled_dQtheta_dtheta_addto_dvec(self->qfn, nS, &self->nA,
                                         -self->gamma, &self->g);
    }
    scaled_vec_addto_dvec(&self->g, self->alpha * delta, &self->qfn->theta);
}

void ResidualGrad_reset(void *self_) {
    ResidualGrad *self = self_;
    // qfn
    ZERO_DV(self->qfn->theta)
    // reset policy: we'll get to it when we get to it
}

AgentVT ResidualGrad_vt = {
    .name = "Residual Gradient",
    .start_ep = ResidualGrad_start_ep,
    .choose_action = ResidualGrad_choose_action,
    .update = ResidualGrad_update,
    .end_ep = ResidualGrad_end_ep,
    .reset = ResidualGrad_reset
};


void make_ResidualGrad(ResidualGrad *self, Environment *env, QFn *qfn,
                       Policy *pi, real alpha, real gamma) {
    self->super.vt = VT_PTR_AMPER ResidualGrad_vt;
    self->super.env = env;
    self->qfn = qfn;
    self->pi = pi;
    self->alpha = alpha;
    self->gamma = gamma;
    make_vec(&self->g, qfn->approx_arch->indim, 0);
    assert(env->SVT_ACCESS action_space_is_fixed);
    assert(env->SVT_ACCESS deterministic_transition);
    make_elem(&self->nA, &env->SVT_ACCESS fixed_action_space);
}

void free_ResidualGrad(ResidualGrad *self) {
    free_vec(&self->g);
    free_elem(&self->nA);
}

// -------------------- NatResidualGrad --------------------------

typedef struct {
    Agent super;
    QFn *qfn;
    Policy *pi;
    Mat Ginv;
    Vec g, v;
    Elem nA;
    real alpha, gamma;
} NatResidualGrad;

void NatResidualGrad_start_ep(void *self_, RngState *rngs, Elem *S_0) {
    NatResidualGrad *self = self_;
    self->pi->VT_ACCESS choose_action(self->pi, rngs, S_0, &self->nA);
}

void NatResidualGrad_end_ep(void *self_, RngState *rngs, uint T) {}

void NatResidualGrad_choose_action(void *self_, RngState *rngs, Elem *S,
                                     Elem *A, uint t) {
    NatResidualGrad *self = self_;
    copy_elem_to_elem(&self->nA,
                      &self->super.env->SVT_ACCESS fixed_action_space, A);
}

void NatResidualGrad_update(void *self_, RngState *rngs,
                         Elem *S, Elem *A, real R, Elem *nS,
                         uint t, bool is_terminal) {
    NatResidualGrad *self = self_;
    real QSA = Qtheta(self->qfn, S, A);
    // delta_t = R_{t+1} + gamma * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
    // g_t = partial Q_theta(S_t, A_t)/partial theta
    //     - gamma*partial Q_theta(S_{t+1}, A_{t+1})/partial theta
    real delta = R - QSA;
    dQtheta_dtheta_writo_vec(self->qfn, S, A, &self->g);
    if (!is_terminal) {
        self->pi->VT_ACCESS choose_action(self->pi, rngs, nS, &self->nA);
        delta += self->gamma * Qtheta(self->qfn, nS, &self->nA);
        scaled_dQtheta_dtheta_addto_dvec(self->qfn, nS, &self->nA,
                                         -self->gamma, &self->g);
    }
    // v = G^{-1}_{t-1} g_t
    dmat_mul_vec_writo_dvec(&self->Ginv, &self->g, &self->v);

    // update G^{-1} with Sherman-Morrison inverse
    real scale = -delta*delta
                / (1 + delta*delta*inner_vec(&self->g, &self->v));
    scaled_self_outer_vec_addto_dmat(&self->v, scale, &self->Ginv);

    // v = G^{-1}_t g_t
    dmat_mul_vec_writo_dvec(&self->Ginv, &self->g, &self->v);

    // update theta
    scaled_vec_addto_dvec(&self->v, self->alpha * delta, &self->qfn->theta);
}

void NatResidualGrad_reset(void *self_) {
    NatResidualGrad *self = self_;
    diag_mat_dmat(&self->Ginv, 1);
    // qfn
    ZERO_DV(self->qfn->theta)
    // reset policy: we'll get to it when we get to it
}

AgentVT NatResidualGrad_vt = {
    .name = "Natural Residual Gradient",
    .start_ep = NatResidualGrad_start_ep,
    .choose_action = NatResidualGrad_choose_action,
    .update = NatResidualGrad_update,
    .end_ep = NatResidualGrad_end_ep,
    .reset = NatResidualGrad_reset
};


void make_NatResidualGrad(NatResidualGrad *self, Environment *env,
                            QFn *qfn, Policy *pi, real alpha, real gamma) {
    self->super.vt = VT_PTR_AMPER NatResidualGrad_vt;
    self->super.env = env;
    self->qfn = qfn;
    self->pi = pi;
    self->alpha = alpha;
    self->gamma = gamma;
    uint d = qfn->approx_arch->indim;
    make_vec(&self->g, d, 0);
    make_vec(&self->v, d, 0);
    make_mat(&self->Ginv, d, d, 0);
    diag_mat_dmat(&self->Ginv, 1);
    assert(env->SVT_ACCESS action_space_is_fixed);
    assert(env->SVT_ACCESS deterministic_transition);
    make_elem(&self->nA, &env->SVT_ACCESS fixed_action_space);
}

void free_NatResidualGrad(NatResidualGrad *self) {
    free_vec(&self->g);
    free_vec(&self->v);
    free_mat(&self->Ginv);
    free_elem(&self->nA);
}

// -------------------- NatResidualGrad_ForgettingG ---------------------

typedef struct {
    Agent super;
    QFn *qfn;
    Policy *pi;
    Mat Ginv;
    Vec g, v;
    Elem nA;
    real alpha, beta, gamma;
} NatResidualGrad_ForgettingG;

void NatResidualGrad_ForgettingG_start_ep(void *self_, RngState *rngs, Elem *S_0) {
    NatResidualGrad_ForgettingG *self = self_;
    self->pi->VT_ACCESS choose_action(self->pi, rngs, S_0, &self->nA);
}

void NatResidualGrad_ForgettingG_end_ep(void *self_, RngState *rngs, uint T) {}

void NatResidualGrad_ForgettingG_choose_action(void *self_, RngState *rngs, Elem *S,
                                     Elem *A, uint t) {
    NatResidualGrad_ForgettingG *self = self_;
    copy_elem_to_elem(&self->nA,
                      &self->super.env->SVT_ACCESS fixed_action_space, A);
}

#define DPR(x) x
int ttt = 0;

void NatResidualGrad_ForgettingG_update(void *self_, RngState *rngs,
                         Elem *S, Elem *A, real R, Elem *nS,
                         uint t, bool is_terminal) {
    DPR(printf("--------- update t=%u ------------\n", ttt++);)
    NatResidualGrad_ForgettingG *self = self_;
    real QSA = Qtheta(self->qfn, S, A);
    // convenience terms
    real beta = self->beta;
    real invcombeta = ((real)1.) / (((real)1.) - beta);
    // delta_t = R_{t+1} + gamma * Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)
    // g_t = partial Q_theta(S_t, A_t)/partial theta
    //     - gamma*partial Q_theta(S_{t+1}, A_{t+1})/partial theta
    real delta = R - QSA;
    dQtheta_dtheta_writo_vec(self->qfn, S, A, &self->g);
    if (!is_terminal) {
        self->pi->VT_ACCESS choose_action(self->pi, rngs, nS, &self->nA);
        delta += self->gamma * Qtheta(self->qfn, nS, &self->nA);
        scaled_dQtheta_dtheta_addto_dvec(self->qfn, nS, &self->nA,
                                         -self->gamma, &self->g);
    }
    // scale G by 1-beta first
    scale_mat(&self->Ginv, invcombeta);

    // v = (1-beta) G^{-1}_{t-1} g_t
    dmat_mul_vec_writo_dvec(&self->Ginv, &self->g, &self->v);

    // update G^{-1} with Sherman-Morrison inverse
    real a = beta * delta * delta;
    // real a = beta * delta * delta;
    real scale = -a / (1 + a*inner_vec(&self->g, &self->v));
    scaled_self_outer_vec_addto_dmat(&self->v, scale, &self->Ginv);


    // v = G^{-1}_t g_t
    dmat_mul_vec_writo_dvec(&self->Ginv, &self->g, &self->v);

    // update theta
    scaled_vec_addto_dvec(&self->v, self->alpha * delta, &self->qfn->theta);

    DPR(
        print_expr(beta, "%.6g");
        print_expr(a, "%.6g");
        printf("g:");
        print_vec(&self->g);
    )
    DPR(
        print_expr(scale, "%.6g");
        printf("new Ginv:");
        print_mat(&self->Ginv);
        printf("new theta:");
        print_vec(&self->qfn->theta);
        getchar();
    )
}

void NatResidualGrad_ForgettingG_reset(void *self_) {
    NatResidualGrad_ForgettingG *self = self_;
    diag_mat_dmat(&self->Ginv, 1);
    // qfn
    ZERO_DV(self->qfn->theta)
    // reset policy: we'll get to it when we get to it
}

AgentVT NatResidualGrad_ForgettingG_vt = {
    .name = "Natural Residual Gradient with Forgetting G",
    .start_ep = NatResidualGrad_ForgettingG_start_ep,
    .choose_action = NatResidualGrad_ForgettingG_choose_action,
    .update = NatResidualGrad_ForgettingG_update,
    .end_ep = NatResidualGrad_ForgettingG_end_ep,
    .reset = NatResidualGrad_ForgettingG_reset
};


void make_NatResidualGrad_ForgettingG(NatResidualGrad_ForgettingG *self, Environment *env,
                            QFn *qfn, Policy *pi, real alpha, real beta, real gamma) {
    self->super.vt = VT_PTR_AMPER NatResidualGrad_ForgettingG_vt;
    self->super.env = env;
    self->qfn = qfn;
    self->pi = pi;
    self->alpha = alpha;
    self->beta = beta;
    self->gamma = gamma;
    uint d = qfn->approx_arch->indim;
    make_vec(&self->g, d, 0);
    make_vec(&self->v, d, 0);
    make_mat(&self->Ginv, d, d, 0);
    diag_mat_dmat(&self->Ginv, 1);
    assert(env->SVT_ACCESS action_space_is_fixed);
    assert(env->SVT_ACCESS deterministic_transition);
    make_elem(&self->nA, &env->SVT_ACCESS fixed_action_space);
}

void free_NatResidualGrad_ForgettingG(NatResidualGrad_ForgettingG *self) {
    free_vec(&self->g);
    free_vec(&self->v);
    free_mat(&self->Ginv);
    free_elem(&self->nA);
}


// -------------------- SarsaLambda ---------------------------

typedef struct {
    Agent super;
    QFn *qfn;
    Policy *pi;
    Vec elig;
    Elem nA;
    real alpha, gamma, lambda;
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
    real QSA = Qtheta(self->qfn, S, A);
    real delta = R - QSA;
    if (!is_terminal) {
        self->pi->VT_ACCESS choose_action(self->pi, rngs, nS, &self->nA);
        delta += self->gamma * Qtheta(self->qfn, nS, &self->nA);
    }
    scale_vec(&self->elig, self->gamma * self->lambda);
    dQtheta_dtheta_addto_dvec(self->qfn, S, A, &self->elig);
    scaled_vec_addto_dvec(&self->elig, self->alpha * delta, &self->qfn->theta);
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
    .name = "Sarsa(lambda)",
    .start_ep = SarsaLambda_start_ep,
    .choose_action = SarsaLambda_choose_action,
    .update = SarsaLambda_update,
    .end_ep = SarsaLambda_end_ep,
    .reset = SarsaLambda_reset
};


void make_SarsaLambda(SarsaLambda *self, Environment *env, QFn *qfn,
                      Policy *pi, real alpha, real gamma, real lambda) {
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

// -------------------- NatSarsaLambda ---------------------------

typedef struct {
    Agent super;
    QFn *qfn;
    Policy *pi;
    Mat Ginv;
    Vec elig, g, v;
    Elem nA;
    real alpha, gamma, lambda;
} NatSarsaLambda;

void NatSarsaLambda_start_ep(void *self_, RngState *rngs, Elem *S_0) {
    NatSarsaLambda *self = self_;
    zero_vec(&self->elig);
    self->pi->VT_ACCESS choose_action(self->pi, rngs, S_0, &self->nA);
}

void NatSarsaLambda_end_ep(void *self_, RngState *rngs, uint T) {}

void NatSarsaLambda_choose_action(void *self_, RngState *rngs,
                                    Elem *S, Elem *A, uint t) {
    NatSarsaLambda *self = self_;
    copy_elem_to_elem(&self->nA,
                      &self->super.env->SVT_ACCESS fixed_action_space, A);
}
uint qq = 0;
void NatSarsaLambda_update(void *self_, RngState *rngs,
                             Elem *S, Elem *A, real R, Elem *nS,
                             uint t, bool is_terminal) {
    NatSarsaLambda *self = self_;
    // calculte delta
    real delta = R - Qtheta(self->qfn, S, A);
    if (!is_terminal) {
        self->pi->VT_ACCESS choose_action(self->pi, rngs, nS, &self->nA);
        delta += self->gamma * Qtheta(self->qfn, nS, &self->nA);
    }



    // calculate g
    dQtheta_dtheta_writo_vec(self->qfn, S, A, &self->g);

        // update e
        scale_vec(&self->elig, self->gamma*self->lambda);
        ADD_DV_TO_DV(self->g, self->elig)


#if 1 // description version if 1, figures version if 0
    scaled_dQtheta_dtheta_addto_dvec(self->qfn, nS, &self->nA, -self->gamma, &self->g);
#endif

    // v = G^{-1} g
    dmat_mul_vec_writo_dvec(&self->Ginv, &self->g, &self->v);

    // update G^{-1} with Sherman-Morrison inverse
    real scale = -delta*delta
                / (1 + delta*delta*inner_vec(&self->g, &self->v));
    scaled_self_outer_vec_addto_dmat(&self->v, scale, &self->Ginv);
    qq++;
    // update theta with natural semigradient
    dmat_mul_vec_writo_dvec(&self->Ginv, &self->elig, &self->v);
    real vmul = self->alpha*delta;
    scaled_vec_addto_dvec(&self->v, vmul, &self->qfn->theta);

    if (0) {
        real normv = sqrtf(inner_vec(&self->v, &self->v));
        real normelig = sqrtf(inner_vec(&self->elig, &self->elig));
        real bilin = inner_vec(&self->v, &self->elig);
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

void reset_NatSarsaLambda(void *self_) {
    NatSarsaLambda *self = self_;
    // internal states
    // g and v are moreso scratch vecs for calculations, no need to reset
    // since they do not meaningfully carry across from timesteps anyway
    ZERO_DV(self->elig)
    diag_mat_dmat(&self->Ginv, 1);
    // qfn
    ZERO_DV(self->qfn->theta)
    // reset policy: we'll get to it when we get to it
}

AgentVT NatSarsaLambda_vt = {
    .name = "Natural Sarsa(lambda)",
    .start_ep = NatSarsaLambda_start_ep,
    .choose_action = NatSarsaLambda_choose_action,
    .update = NatSarsaLambda_update,
    .end_ep = NatSarsaLambda_end_ep,
    .reset = reset_NatSarsaLambda
};

void make_NatSarsaLambda(NatSarsaLambda *self, Environment *env, QFn *qfn,
                      Policy *pi, real alpha, real gamma, real lambda) {
    self->super.vt = VT_PTR_AMPER NatSarsaLambda_vt;
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

void free_NatSarsaLambda(NatSarsaLambda *self) {
    free_vec(&self->elig);
    free_vec(&self->g);
    free_vec(&self->v);
    free_mat(&self->Ginv);
    free_elem(&self->nA);
}
