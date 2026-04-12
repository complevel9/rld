
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
    case 0: {
        ResidualGrad *rg = custom_malloc(sizeof *rg);
        make_ResidualGrad(rg, *env, qfn, (Policy*)epgreedy,
            hpt[1].r, hpt[2].r);
        *ag = (Agent*) rg;
    } break;
    case 1: {
    } break;
    case 2: {
        SarsaLambda *sl = custom_malloc(sizeof *sl);
        make_SarsaLambda(sl, *env, qfn, (Policy*)epgreedy,
            hpt[1].r, hpt[2].r, hpt[3].r);
        *ag = (Agent*) sl;
    } break;
    case 3: {
        QTNatSarsaLambda *qnsl = custom_malloc(sizeof *qnsl);
        make_QTNatSarsaLambda(qnsl, *env, qfn, (Policy*)epgreedy,
            hpt[1].r, hpt[2].r, hpt[3].r);
        *ag = (Agent*) qnsl;
    } break;
    default:
        abort();
    }
}

void free_mcar_experiment(Environment *env_, Agent *ag_, uint agi) {
    EpGreedy *epgreedy;
    QFn *qfn;
    switch (agi) {
    case 0: {
        ResidualGrad *ag = (ResidualGrad*) ag_;
        epgreedy = (EpGreedy*) ag->pi;
        qfn = (QFn*) ag->qfn;
        free_ResidualGrad(ag);
    } break;
    // case 1: {
    // } break;
    case 2: {
        SarsaLambda *ag = (SarsaLambda*) ag_;
        epgreedy = (EpGreedy*) ag->pi;
        qfn = (QFn*) ag->qfn;
        free_SarsaLambda(ag);
    } break;
    case 3: {
        QTNatSarsaLambda *ag = (QTNatSarsaLambda*) ag_;
        epgreedy = (EpGreedy*) ag->pi;
        qfn = (QFn*) ag->qfn;
        free_QTNatSarsaLambda(ag);
    } break;
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

void mcar_exp_fb_vis(Environment *env_, Agent *ag_, Elem *S, Elem *A, float R,
                     Elem *nS, uint t, void *ep_) {
    // MountainCar *env = (void*)env_;
    QTNatSarsaLambda *ag = (void*)ag_;
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
        al_build_transform(&Tr_text, 0, 0, 2, 2, 0);

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
    al_draw_textf(font, WHITE, 15, 15, 0, "Episode: %u", ep);
    al_draw_textf(font, WHITE, 15, 25, 0, "Time: %u", t);

    // static float val_max = 8.f, val_min = -40.f;
    static float val_max = 0.f, val_min = -1.f;
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
}



Experiment mcar_exp = {
    .make_exp = make_mcar_experiment,
    .free_exp = free_mcar_experiment,
    .visfn = mcar_exp_fb_vis,
    .steps_per_vis = 2,
    .nagents = 1, // <-------------------------------------- testing
    .nepisodes = 15,
    .ntrials = 10,

    .nepisodes_hpts = 10,
    .ntrials_hpts   = 50,
    .nsamples_hpts  = 1000,
    .stuck_hptskip_times = MC_STOPSHORT_MAX_T,
    .hptss = (HParamsTupleSearch[]) {
        { // rg
            .nhparams = 3,
            .hparam_search = (HParamSearch[]) {
                {"eps", rand_exprange_writo, {{.range={1e-5, 0.3}}, 'r'}},
                {"alp", rand_exprange_writo, {{.range={1e-4, 0.1}}, 'r'}},
                {"gam", rand_1m_exprange_writo, {{.range={5e-4, 0.9}}, 'r'}},
            }
        },
        { // nat rg
            .nhparams = 3,
            .hparam_search = (HParamSearch[]) {
                {"eps", rand_exprange_writo, {{.range={1e-5, 0.3}}, 'r'}},
                {"alp", rand_exprange_writo, {{.range={1e-4, 0.1}}, 'r'}},
                {"gam", rand_1m_exprange_writo, {{.range={5e-4, 0.9}}, 'r'}},
            }
        },
        { // sl
            .nhparams = 4,
            .hparam_search = (HParamSearch[]) {
                {"eps", rand_exprange_writo, {{.range={1e-5, 0.3}}, 'r'}},
                {"alp", rand_exprange_writo, {{.range={1e-4, 0.1}}, 'r'}},
                {"gam", rand_1m_exprange_writo, {{.range={5e-4, 0.9}}, 'r'}},
                {"lam", rand_1m_exprange_writo, {{.range={1e-4, 0.9}}, 'r'}},
            }
        },
        { // qt nat sl
            .nhparams = 4,
            .hparam_search = (HParamSearch[]) {
                {"eps", rand_exprange_writo, {{.range={1e-5, 0.3}}, 'r'}},
                {"alp", rand_exprange_writo, {{.range={5e-4, 0.1}}, 'r'}},
                {"gam", rand_1m_exprange_writo, {{.range={1e-4, 0.9}}, 'r'}},
                {"lam", rand_1m_exprange_writo, {{.range={1e-4, 0.9}}, 'r'}},
            }
        },
    },
    .best_hpt_for_ag = (HParam*[]) {
        (HParam[]) // rg;
        {{.r=0.0001}, {.r=0.002}, {.r=0.995}},
        (HParam[]) // nat rg;
        {{.r=0.0001}, {.r=0.002}, {.r=0.995}},
        (HParam[]) // sl; mean ret=-195.18
        {{.r=0.00005241}, {.r=0.06805436}, {.r=0.99874651}, {.r=0.38837796}},
        (HParam[]) // qtnatsl; mean ret=-167.16
        {{.r=0.00008165}, {.r=0.07211637}, {.r=0.99987674}, {.r=0.61488545}},
    }
};
