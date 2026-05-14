#define MCAR_EXP_GAMMA 0.995

void make_mcar_experiment(Environment **env, Agent **ag, uint agi,
                          HParam *hpt, bool search) {
    MountainCar *mcar = custom_malloc(sizeof *mcar);
    make_MountainCar(mcar, search ? MC_FL_STOPSHORT : 0);
    *env = (Environment*) mcar;

    uint *fourier_orders = custom_malloc(2 * sizeof(uint));
    fourier_orders[0] = 3;
    fourier_orders[1] = 3;
    FourierBasis *sfmap = custom_malloc(sizeof *sfmap);
    make_FourierBasis(sfmap, &(*env)->SVT_ACCESS state_space, fourier_orders);

    OneHot *afmap = custom_malloc(sizeof *afmap);
    make_OneHot(afmap, &(*env)->SVT_ACCESS fixed_action_space, 0);

    FlatOuterProduct *safmap = custom_malloc(sizeof *safmap);
    make_FlatOuterProduct(safmap, sfmap, afmap, FLATOUTERPROD_FL_FLIPORDER);

    Linear *lin = custom_malloc(sizeof *lin);
    make_Linear(lin, ((SAFeatureMap*)safmap)->outdim);

    QFn *qfn = custom_malloc(sizeof *qfn);
    make_QFn(qfn, (SAFeatureMap*)safmap, (SmoothParametricFn*)lin);

    EpGreedy *epgreedy = custom_malloc(sizeof(EpGreedy));
    make_EpGreedy(epgreedy, (*env), qfn, hpt[0].r);

    switch (agi) {
    #define AGENT(num, name, ...) \
        case num: { \
            name *the_ag = custom_malloc(sizeof *the_ag); \
            make_ ## name (the_ag, *env, qfn, (Policy*)epgreedy, __VA_ARGS__); \
            *ag = (Agent*) the_ag; \
        } break;

                                             // alpha     beta      gamma           lambda    aem  errclip
        AGENT(0, ResidualGrad,                  hpt[1].r,           MCAR_EXP_GAMMA)
        AGENT(1, NatResidualGrad,               hpt[1].r,           MCAR_EXP_GAMMA)
        AGENT(2, NatResidualGrad_ForgetfulG,    hpt[1].r, hpt[2].r, MCAR_EXP_GAMMA,           0,   hpt[3].r)
        AGENT(3, NatResidualGrad_ForgetfulG,    hpt[1].r, hpt[2].r, MCAR_EXP_GAMMA,           1,   0)

        AGENT(4, SarsaLambda,                   hpt[1].r,           MCAR_EXP_GAMMA, hpt[2].r)
        AGENT(5, NatSarsaLambda,                hpt[1].r,           MCAR_EXP_GAMMA, hpt[2].r)
        AGENT(6, NatSarsaLambda_ForgetfulG,     hpt[1].r, hpt[2].r, MCAR_EXP_GAMMA, hpt[3].r, 0,   hpt[4].r)
        AGENT(7, NatSarsaLambda_ForgetfulG,     hpt[1].r, hpt[2].r, MCAR_EXP_GAMMA, hpt[3].r, 1,   0)

    #undef AGENT
    default:
        abort();
    }
}

void free_mcar_experiment(Environment *env_, Agent *ag_, uint agi) {
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

        EXTRACT_EPGREEDY_QFN(4, SarsaLambda)
        EXTRACT_EPGREEDY_QFN(5, NatSarsaLambda)
        EXTRACT_EPGREEDY_QFN(6, NatSarsaLambda_ForgetfulG)
        EXTRACT_EPGREEDY_QFN(7, NatSarsaLambda_ForgetfulG)

    #undef EXTRACT_EPGREEDY_QFN
    default:
        abort();
    }
    free(ag_);

    free_EpGreedy(epgreedy);
    free(epgreedy);

    FlatOuterProduct *safmap = (FlatOuterProduct*) qfn->sa_feamap;
        FourierBasis *sfmap = (FourierBasis*)safmap->super.s_feamap;
            free(sfmap->orders);
        free_FourierBasis(sfmap);
        free(sfmap);
        free_OneHot((OneHot*)safmap->super.a_feamap);
        free(safmap->super.a_feamap);
    free_FlatOuterProduct(safmap);
    free(safmap);

    free_Linear((Linear*)qfn->approx_arch);
    free(qfn->approx_arch);

    free_QFn(qfn);
    free(qfn);

    free_MountainCar((MountainCar*)env_);
    free(env_);
}

// -------------------- Visualization --------------------------

void mcar_exp_fb_vis(Environment *env_, Agent *ag_, Elem *S, Elem *A, float R,
                     Elem *nS, uint t, void *ep_) {
    // MountainCar *env = (void*)env_;
    NatSarsaLambda *ag = (void*)ag_; // this is a hack
    uint ep = *(uint*)ep_;
    QFn *q = ag->qfn;

    static bool static_initialized = false;
    #define CURVE_NPOINTS 40
    #define QBMP_RES 24
    #define QBMP_FORMAT ALLEGRO_PIXEL_FORMAT_XRGB_8888
    static ALLEGRO_VERTEX curve[CURVE_NPOINTS];
    static ALLEGRO_TRANSFORM Tr_quadrant[4];
    static ALLEGRO_TRANSFORM Tr_text;
    static ALLEGRO_BITMAP *qbmp[3];
    if (!static_initialized) {
        for (uint i = 0; i < CURVE_NPOINTS; i++) {
            float fracx = i / (CURVE_NPOINTS - 1.f); // [0, 1]
            float x = MC_POS_FROM_UNIT(fracx);
            float fracy = 0.5f + 0.4f*sinf(3*x);
            curve[i].x = fracx;
            curve[i].y = fracy;
            curve[i].z = 0;
            curve[i].color = GREEN;
        }
        for (uint qy = 0; qy < 2; qy++) {
            for (uint qx = 0; qx < 2; qx++) {
                float qscale = !qx && !qy ? 1 : 1./QBMP_RES;
                al_build_transform(&Tr_quadrant[qy*2+qx],
                                   DWIDTH/2*qx+10, DHEIGHT/2*(qy+1)-10,
                                   qscale*(DWIDTH/2-20),
                                   qscale*(20-DHEIGHT/2),
                                   0);
            }
        }
        al_build_transform(&Tr_text, 0, 0, 1, 1, 0);

        al_set_new_bitmap_format(QBMP_FORMAT);
        al_set_new_bitmap_flags(ALLEGRO_NO_PRESERVE_TEXTURE
                                | ALLEGRO_MAG_LINEAR
                                | ALLEGRO_VIDEO_BITMAP);
        for (uint i = 0; i < 3; i++) {
            qbmp[i] = al_create_bitmap(QBMP_RES, QBMP_RES);
        }
        static_initialized = true;
    }
    al_clear_to_color(BLACK);

    al_use_transform(&Tr_quadrant[0]);
    al_draw_rectangle(0, 0, 1, 1, WHITE, 0);
    al_draw_prim(curve, NULL, NULL, 0, CURVE_NPOINTS,
                 ALLEGRO_PRIM_LINE_STRIP);

    float car_ux = MC_POS_TO_UNIT(S->MC_POS);
    float car_ux_curve_f = rmin(car_ux * (CURVE_NPOINTS-1.f),
                                CURVE_NPOINTS-1.f);
    uint car_ux_curve_i = car_ux_curve_f;
    float car_ux_curve_frac = car_ux_curve_f - car_ux_curve_i;
    float car_uy = curve[car_ux_curve_i].y  *(1.f - car_ux_curve_frac)
                + curve[car_ux_curve_i+1].y*car_ux_curve_frac;
    int dir = ((int)A->MC_ACCEL) - 1;
    al_draw_circle(car_ux, car_uy, 0.02f, WHITE, 0);
    al_draw_line(car_ux, car_uy, car_ux + 0.08f*dir, car_uy, YELLOW, 0);

    float car_uspeed = MC_SPEED_TO_UNIT(S->MC_SPEED);

    // al_use_transform(&ID_TRANS);
    al_use_transform(&Tr_text);
    al_draw_textf(font, WHITE, 15, 15, 0, "%s", ag_->VT_ACCESS name);
    al_draw_textf(font, WHITE, 15, 25, 0, "Episode: %u", ep);
    al_draw_textf(font, WHITE, 15, 35, 0, "Time: %u", t);

    // static float val_max = 8.f, val_min = -40.f;
    static float val_max = 0.f, val_min = -0.1f;
    al_set_target_backbuffer(display);
    for (uint i = 0; i < 3; i++) {
        ALLEGRO_LOCKED_REGION *bmr = al_lock_bitmap(qbmp[i], QBMP_FORMAT,
          ALLEGRO_LOCK_WRITEONLY);

        uint32_t *pix = bmr->data;
        for (uint y = 0; y < QBMP_RES; y++) {
            for (uint x = 0; x < QBMP_RES; x++) {
                float val = MC_Qtheta(q,
                  MC_POS_FROM_UNIT(x/(QBMP_RES-1.f)),
                  MC_SPEED_FROM_UNIT(y/(QBMP_RES-1.f)), i);
                if (val > val_max) val_max = val;
                if (val < val_min) val_min = val;
                float uval = (val - val_min) / (val_max - val_min);
                if (uval < 0.f || uval > 1.f)
                    printf("uval = %.6g out of range\n", uval);
                *pix = cmap_magma(uval);
                pix += 1;
            }
            pix -= QBMP_RES;
            pix = (uint32_t*)(bmr->pitch + (uchar*)pix);
        }
        al_unlock_bitmap(qbmp[i]);

        ALLEGRO_TRANSFORM *Trq = &Tr_quadrant[((uint[]){2, 1, 3})[i]];
        al_use_transform(Trq);
        al_draw_bitmap(qbmp[i], 0, 0, 0);
        // ALLEGRO_COLOR ccol = i == A->MC_ACCEL ? CYAN : WHITE;
        if (i == A->MC_ACCEL) {
            // al_draw_filled_circle(car_ux*QBMP_RES, car_uspeed*QBMP_RES, 0.7f,
            //     WHITE);
            // al_draw_filled_circle(car_ux*QBMP_RES, car_uspeed*QBMP_RES, 0.5f,
            //     RED);
            al_draw_circle(car_ux*QBMP_RES, car_uspeed*QBMP_RES, 0.5f,
                WHITE, 0);
            al_draw_line(car_ux*QBMP_RES, car_uspeed*QBMP_RES,
                         car_ux*QBMP_RES, car_uspeed*QBMP_RES+3*dir,
                         YELLOW, 0);
        } else {
            // al_draw_circle(car_ux*QBMP_RES, car_uspeed*QBMP_RES, 0.2f,
            //     WHITE, 0);
        }
        al_draw_rectangle(0, 0, QBMP_RES, QBMP_RES, WHITE, 0);
    }

    al_flip_display();

    // print_mat(&ag->Ginv);

}

Experiment mcar_exp;
void print_mcar_ret(Environment *env_, Agent *ag_, Elem *S, Elem *A, float R,
                     Elem *nS, uint t, void *ep_) {
    uint ep = *(uint*)ep_;
    static real ret = 0;
    ret += R;
    if (MountainCar_is_terminal(env_, nS, t)) {
        fprintf(fp_ret, "%.6g%c", ret,
          ep == mcar_exp.nepisodes - 1 ? '\n' : ',');
        ret = 0;
    }
}

// macros for convenience, since the ranges are similar anyway
#define SEARCH_EPSILON {"eps", rand_exprange_writo, {{.range={0.002, 0.2}}, 'r'}}
#define SEARCH_ALPHA   {"alp", rand_exprange_writo, {{.range={1e-4, 0.2}}, 'r'}}
#define SEARCH_BETA    {"bet", rand_exprange_writo, {{.range={1e-4, 0.2}}, 'r'}}
// #define SEARCH_GAMMA   {"gam", rand_1m_exprange_writo, {{.range={1e-4, 1.0}}, 'r'}}
#define SEARCH_LAMBDA  {"lam", rand_1m_exprange_writo, {{.range={1e-3, 1.0}}, 'r'}}
#define SEARCH_ERRCLIP {"erc", rand_exprange_writo, {{.range={0.3, 5.0}}, 'r'}}

Experiment mcar_exp = {
    .make_exp = make_mcar_experiment,
    .free_exp = free_mcar_experiment,

    #if 1 // visual demonstration
        .visfn = mcar_exp_fb_vis,
        .steps_per_vis = 1,
        .flags = EXPERIMENT_VIS_REQUIRE_ALLEGRO
            // | EXPERIMENT_RUN_GETCH
        ,
    #else // only print undiscounted return to file
        .visfn = print_mcar_ret,
        .steps_per_vis = 1,
        .flags = 0,
    #endif

    .nagents = 4,
    .nepisodes = 15,
    .ntrials = 3,
    .stuck_timesteps = MC_MAX_T,

    .nsamples_hpts  = 40000,
    .nepisodes_hpts = 20,
    .ntrials_hpts   = 1,
    .stuck_timesteps_hpt = MC_STOPSHORT_MAX_T,
    .hptss = (HParamsTupleSearch[]) {
        { // rg
            .nhparams = 2,
            .hparam_search = (HParamSearch[]) {
                SEARCH_EPSILON, SEARCH_ALPHA,
            }
        },
        { // nat rg
            .nhparams = 2,
            .hparam_search = (HParamSearch[]) {
                SEARCH_EPSILON, SEARCH_ALPHA,
            }
        },
        { // nat rg-fg w/ errclip
            .nhparams = 4,
            .hparam_search = (HParamSearch[]) {
                SEARCH_EPSILON, SEARCH_ALPHA, SEARCH_BETA, SEARCH_ERRCLIP,
            }
        },
        { // nat rg-faeg
            .nhparams = 3,
            .hparam_search = (HParamSearch[]) {
                SEARCH_EPSILON, SEARCH_ALPHA, SEARCH_BETA,
            }
        },
        // ------------ Sarsa(lambda) based algorithms
        { // sl
            .nhparams = 3,
            .hparam_search = (HParamSearch[]) {
                SEARCH_EPSILON, SEARCH_ALPHA, SEARCH_LAMBDA,
            }
        },
        { // nat sl
            .nhparams = 3,
            .hparam_search = (HParamSearch[]) {
                SEARCH_EPSILON, SEARCH_ALPHA, SEARCH_LAMBDA,
            }
        },
        { // nat sl-fg w/ errclip
            .nhparams = 5,
            .hparam_search = (HParamSearch[]) {
                SEARCH_EPSILON, SEARCH_ALPHA, SEARCH_BETA, SEARCH_LAMBDA, SEARCH_ERRCLIP,
            }
        },
        { // nat sl-faeg
            .nhparams = 4,
            .hparam_search = (HParamSearch[]) {
                SEARCH_EPSILON, SEARCH_ALPHA, SEARCH_BETA, SEARCH_LAMBDA,
            }
        },
    },
    .best_hpt_for_ag = (HParam*[]) {
        // epsilon        alpha            beta              lambda            errclip
        (HParam[]) // rg; mean ret=-249.42
        {{.r=0.00022221}, {.r=0.16062324}},
        (HParam[]) // nat rg; mean ret=-263.27
        {{.r=0.00001640}, {.r=0.11993723}},
        (HParam[]) // nat rg-fg w/ errclip; mean ret=
        {{.r=0.02}, {.r=0.005}, {.r=0.0001}, {.r=0.5}},
        (HParam[]) // nat rg-faeg; mean ret=
        {{.r=0.02}, {.r=0.005}, {.r=0.0001}},


        (HParam[]) // sl; mean ret=-193.29
        {{.r=0.00047899}, {.r=0.09324095},                   {.r=0.22648823}},
        (HParam[]) // nat sl; mean ret=-142.66
        {{.r=0.00049613}, {.r=0.17115353},                   {.r=0.00953287}},
        (HParam[]) // nat sl-fg w/ errclip; mean ret=
        {{.r=0.02}, {.r=0.002}, {.r=0.0001}, {.r=0.95}, {.r=0.5}},
        (HParam[]) // nat sl-faeg; mean ret=
        {{.r=0.02}, {.r=0.002}, {.r=0.0001}, {.r=0.95}},
    }
};
