#define FLIP_GREEDY_EPSILON 0.02

typedef SAFeatureMap FlipFMap;
void FlipFMap_map(void *self_, Elem *S, Elem *A, Vec *dest) {
    dest->data.dense[0] = A->x[0].i == 0;
    dest->data.dense[1] = A->x[0].i == 1;
}

void FlipFMap_prealloc_out_vec(void *self_, Vec *dest) {
    make_vec(dest, 2, 0);
}

SAFeatureMapVT FlipFMap_vt = {
    .map_writo_vec = FlipFMap_map,
    .prealloc_out_vec = FlipFMap_prealloc_out_vec,
};

void make_FlipFMap(FlipFMap *self) {
    ((SAFeatureMap*)self)->vt = VT_PTR_AMPER FlipFMap_vt;
    ((SAFeatureMap*)self)->outdim = 2;
}

void free_FlipFMap(FlipFMap *self) {}


void make_flip_experiment(Environment **env, Agent **ag, uint agi,
                          HParam *hpt, bool search) {
    Flip *flip = custom_malloc(sizeof *flip);
    make_Flip(flip);
    *env = (Environment*) flip;

    FlipFMap *safmap = custom_malloc(sizeof *safmap);
    make_FlipFMap(safmap);

    Linear *lin = custom_malloc(sizeof *lin);
    make_Linear(lin, ((SAFeatureMap*)safmap)->outdim);

    QFn *qfn = custom_malloc(sizeof *qfn);
    make_QFn(qfn, (SAFeatureMap*)safmap, (SmoothParametricFn*)lin);

    EpGreedy *epgreedy = custom_malloc(sizeof(EpGreedy));
    make_EpGreedy(epgreedy, *env, qfn, FLIP_GREEDY_EPSILON);

    switch (agi) {
    case 0: {
        ResidualGrad *rg = custom_malloc(sizeof *rg);
        make_ResidualGrad(rg, *env, qfn, (Policy*) epgreedy,
            hpt[0].r, 0.f); // alpha, gamma
        *ag = (Agent*) rg;
    } break;
    case 1: {
        NatResidualGrad *natrg = custom_malloc(sizeof *natrg);
        make_NatResidualGrad(natrg, *env, qfn, (Policy*) epgreedy,
            hpt[0].r, 0.f); // alpha, gamma
        *ag = (Agent*) natrg;
    } break;
    case 2: {
        NatResidualGrad_ForgetfulG *natrg_fg = custom_malloc(sizeof *natrg_fg);
        make_NatResidualGrad_ForgetfulG(natrg_fg, *env, qfn, (Policy*) epgreedy,
            hpt[0].r, hpt[1].r, 0.f, false, 0.5f); // alpha, beta, gamma, no absolute error G, errclip
        *ag = (Agent*) natrg_fg;
    } break;
    case 3: {
        NatResidualGrad_ForgetfulG *natrg_faeg = custom_malloc(sizeof *natrg_faeg);
        make_NatResidualGrad_ForgetfulG(natrg_faeg, *env, qfn, (Policy*) epgreedy,
            hpt[0].r, hpt[1].r, 0.f, true, 0.f); // alpha, beta, gamma, absolute error G, errclip
        *ag = (Agent*) natrg_faeg;
    } break;
    // case 2: {
        // xx *yy = custom_malloc(sizeof *yy);
        // make_xx(yy, *env, qfn, (Policy*) epgreedy,);
        // *ag = (Agent*) yy;
    // } break;
    // case 2: {
        // xx *yy = custom_malloc(sizeof *yy);
        // make_xx(yy, *env, qfn, (Policy*) epgreedy,);
        // *ag = (Agent*) yy;
    // } break;

    default:
        abort();
    }
}

Experiment flip_exp;
void print_flip(Environment *env_, Agent *ag_, Elem *S, Elem *A, float R,
                  Elem *nS, uint t, void *ep_) {
    NatResidualGrad_ForgetfulG *ag = (void*)ag_; // this is a hack to get qfn
    uint ep = *(uint*)ep_;
    QFn *q = ag->qfn;

    // printf("theta: ");
    // print_vec(&q->theta);
    // print_mat(&ag->Ginv);
    fprintf(fp_theta, "%.6g,%.6g%c",
      q->theta.data.dense[0],
      q->theta.data.dense[1],
      ep == flip_exp.nepisodes - 1 ? '\n' : ',');
    fprintf(fp_ret, "%.6g%c", R,
      ep == flip_exp.nepisodes - 1 ? '\n' : ',');
    // more hacks - checks name for 'N'(atural), Ginvs has same struct offsets
    if (ag_->VT_ACCESS name[0] == 'N') {
        fprintf(fp_metric, "%.6g,%.6g%c",
          ag->Ginv.data.dense[0],
          ag->Ginv.data.dense[3],
          ep == flip_exp.nepisodes - 1 ? '\n' : ',');
    }

    // getchar();

    return;
}

void free_flip_experiment(Environment *env_, Agent *ag_, uint agi) {
    EpGreedy *epgreedy;
    QFn *qfn;
    switch (agi) {
    #define EXTRACT_EPGREEDY_QFN(num, name) \
        case num: { \
                name *ag = (name*) ag_; \
                epgreedy = (EpGreedy*) ag->pi; \
                qfn = (QFn*) ag->qfn; \
                free_ ## name (ag); \
            } break;
        EXTRACT_EPGREEDY_QFN(0, ResidualGrad)
        EXTRACT_EPGREEDY_QFN(1, NatResidualGrad)
        EXTRACT_EPGREEDY_QFN(2, NatResidualGrad_ForgetfulG)
        EXTRACT_EPGREEDY_QFN(3, NatResidualGrad_ForgetfulG)
    default:
        abort();
    }

    free(ag_);

    free_EpGreedy(epgreedy);
    free(epgreedy);

    free_Linear((Linear*) qfn->approx_arch);
    free(qfn->approx_arch);

    free_FlipFMap((FlipFMap*) qfn->sa_feamap);
    free(qfn->sa_feamap);

    free_QFn(qfn);
    free(qfn);

    free_Flip((Flip*) env_);
    free(env_);
}

Experiment flip_exp = {
    .make_exp = make_flip_experiment,
    .free_exp = free_flip_experiment,
    .visfn = print_flip,
    .steps_per_vis = 1,
    .flags = 0,

    .nagents = 3,
    .nepisodes = 8000,
    .ntrials = 20,
    .stuck_timesteps = 2,

    .best_hpt_for_ag = (HParam*[]) {
        // alpha      // beta      // lambda
        (HParam[]) // rg;
        {{.r=0.01}},
        (HParam[]) // natrg;
        {{.r=0.2}},
        (HParam[]) // natrg_fg;
        {{.r=0.01},    {.r=0.01}},
        (HParam[]) // natrg_faeg;
        {{.r=0.01},    {.r=0.01}},
    }
};
