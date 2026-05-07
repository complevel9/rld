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
        QTNatResidualGrad *qtnatrg = custom_malloc(sizeof *qtnatrg);
        make_QTNatResidualGrad(qtnatrg, *env, qfn, (Policy*) epgreedy,
            hpt[0].r, 0.f); // alpha, gamma
        *ag = (Agent*) qtnatrg;
    } break;
    case 2: {
        QTNatResidualGrad_FG *qtnatrg_fg = custom_malloc(sizeof *qtnatrg_fg);
        make_QTNatResidualGrad_FG(qtnatrg_fg, *env, qfn, (Policy*) epgreedy,
            hpt[0].r, hpt[1].r, 0.f); // alpha, beta, gamma
        *ag = (Agent*) qtnatrg_fg;
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

void print_Q_flip(Environment *env_, Agent *ag_, Elem *S, Elem *A, float R,
                  Elem *nS, uint t, void *ep_) {
    QTNatResidualGrad_FG *ag = (void*)ag_; // this is a hack to get qfn
    uint ep = *(uint*)ep_;
    QFn *q = ag->qfn;

    // printf("theta: ");
    // print_vec(&q->theta);
    // print_mat(&ag->Ginv);
    // fprintf(stdout, "%.6g,%.6g%c",
    //   q->theta.data.dense[0],
    //   q->theta.data.dense[1],
    //   ep < 1000-1 ? ',' : '\n');
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
    // case 0: {
    //     ResidualGrad *ag = (ResidualGrad*) ag_;
    //     epgreedy = (EpGreedy*) ag->pi;
    //     qfn = (QFn*) ag->qfn;
    //     free_ResidualGrad(ag);
    // } break;
    // case 1: {
    //     QTNatResidualGrad *ag = (QTNatResidualGrad*) ag_;
    //     epgreedy = (EpGreedy*) ag->pi;
    //     qfn = (QFn*) ag->qfn;
    //     free_QTNatResidualGrad(ag);
    // } break;
    // case 2: {
    //     ResidualGrad *ag = (ResidualGrad*) ag_;
    //     epgreedy = (EpGreedy*) ag->pi;
    //     qfn = (QFn*) ag->qfn;
    //     free_ResidualGrad(ag);
    // } break;
        EXTRACT_EPGREEDY_QFN(0, ResidualGrad)
        EXTRACT_EPGREEDY_QFN(1, QTNatResidualGrad)
        EXTRACT_EPGREEDY_QFN(2, QTNatResidualGrad_FG)
    // case 1: {
    //     ResidualGrad *ag = (ResidualGrad*) ag_;
    //     epgreedy = (EpGreedy*) ag->pi;
    //     qfn = (QFn*) ag->qfn;
    //     free_ResidualGrad(ag);
    // } break;
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
    .visfn = print_Q_flip,
    .steps_per_vis = 1,
    .flags = 0,

    .nagents = 3,
    .nepisodes = 8000,
    .ntrials = 1,
    .stuck_timesteps = MC_MAX_T,

    .best_hpt_for_ag = (HParam*[]) {
        // alpha      // beta      // lambda
        (HParam[]) // rg;
        {{.r=0.01}},
        (HParam[]) // qtnatrg;
        {{.r=0.2}},
        (HParam[]) // qtnatrg_fg;
        {{.r=0.01},    {.r=0.005}},
    }
};
