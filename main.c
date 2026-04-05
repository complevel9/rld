#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>

// parallel hyperparam search
#include <omp.h>

// for visualization
#include <allegro5/allegro.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_opengl.h>

// this is actually monolithic, the translation unit is just split up.

#include "util.c" // misc utilities: OOP macros, math, RNG

#include "vecmat.c"
#include "space.c"
#include "featuremap.c"
#include "fnapprox.c"


// RL stuff
// avoid any allocation during experiment

#include "actionvalue.c"
#include "environment.c"
#include "policy.c"
#include "agent.c"

#include "test.c"

// -------------------- Param search ---------------------------

typedef struct {
    uint i;
    real r;
} HParam;

typedef void (*SetSampler)(RngState *rngs, SimpleSet, HParam *out);

void rand_exprange_writo(RngState *rngs, SimpleSet s, HParam *out) {
    Range r = s.as.range;
    out->r = rand_exprange(rngs, r.a, r.b);
}

void rand_1m_exprange_writo(RngState *rngs, SimpleSet s, HParam *out) {
    Range r = s.as.range;
    out->r = 1.f - rand_exprange(rngs, r.a, r.b);
}

typedef struct {
    SetSampler sampler;
    char *name;
    SimpleSet sset;
} HParamSearch;

typedef struct {
    uint nhparams;
    HParamSearch *hparam_search;
} HParamsTupleSearch;

void sample_hparams_tuple(RngState *rngs, HParamsTupleSearch *hpts,
                          HParam *out) {
    for (uint i = 0; i < hpts->nhparams; i++) {
        HParamSearch hps = hpts->hparam_search[i];
        hps.sampler(rngs, hps.sset, &out[i]);
    }
}

void print_hparams_tuple(HParamsTupleSearch *hpts, HParam *hpt) {
    for (uint i = 0; i < hpts->nhparams; i++) {
        HParamSearch hps = hpts->hparam_search[i];
        if (hps.sset.type=='r')
            printf("%s=%.8f ", hps.name, hpt[i].r);
        else if (hps.sset.type=='d')
            printf("%s=%u ", hps.name, hpt[i].i);
        else {
            // printf("abort tid=%u\n", hps.uthrash);
            abort();
        }
    }
    putchar('\n');
}


// -------------------- ALLEGRO stuff --------------------------

#define DWIDTH 800
#define DHEIGHT 800
bool allegro_active = false;
ALLEGRO_EVENT_QUEUE *event_queue = NULL;
ALLEGRO_DISPLAY *display = NULL;
ALLEGRO_FONT *font = NULL;
const float FPS = 60.f;

static ALLEGRO_COLOR BLACK, RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA,
    WHITE;
static ALLEGRO_TRANSFORM ID_TRANS;
void init_allegro() {
    if (allegro_active)
        return;
    #define do_die(do,die_msg) if (!(do)) {fputs(die_msg, stderr); exit(1);}
    #define init_die(do,die_msg) do_die((do), "Couldn't init " die_msg "!")
    init_die(al_init(), "allegro")
    init_die(al_init_primitives_addon(), "primitives addon")
    init_die(al_install_keyboard(), "keyboard")

    event_queue = al_create_event_queue();
    init_die(event_queue, "event_queue");

    display = al_create_display(DWIDTH, DHEIGHT);
    init_die(display, "display")
    al_set_window_title(display, "woah salsa");

    font = al_create_builtin_font();
    init_die(font, "font")

    al_register_event_source(event_queue, al_get_keyboard_event_source());
    al_register_event_source(event_queue,
      al_get_display_event_source(display));

    // allegro related utilities
    BLACK   = al_map_rgb(0, 0, 0);
    RED     = al_map_rgb(255, 0, 0);
    GREEN   = al_map_rgb(0, 255, 0);
    BLUE    = al_map_rgb(0, 0, 255);
    YELLOW  = al_map_rgb(255, 255, 0);
    CYAN    = al_map_rgb(0, 255, 255);
    MAGENTA = al_map_rgb(255, 0, 255);
    WHITE   = al_map_rgb(255, 255, 255);

    al_identity_transform(&ID_TRANS);
    // getchar();
    allegro_active = true;
}

typedef void (*Visfn)(Environment *env, Agent *ag, Elem *S, Elem *A, float R,
                      Elem *nS, uint t, void *visinfo);

void destroy_allegro() {
    if (!allegro_active)
        return;

    al_destroy_event_queue(event_queue);
    al_destroy_display(display);
    al_destroy_font(font);

    allegro_active = false;
}

#include "colormap.c"

// -------------------- experinments ---------------------------


#define RUN_GETCH 1

uint run_episode(RngState *rngs, Environment *env, Agent *ag, real *outG,
                 uint steps_per_vis, Visfn vis, void *visinfo,
                 uint flags) {
    Elem S, nS, A, *Sp = &S, *nSp = &nS, *tmp;
    make_elem(Sp,  &env->SVT_ACCESS state_space);
    make_elem(nSp, &env->SVT_ACCESS state_space);
    make_elem(&A,  &env->SVT_ACCESS fixed_action_space);

    assert(env->SVT_ACCESS action_space_is_fixed);
    assert(env->SVT_ACCESS action_space_is_discrete);

    env->VT_ACCESS start_state(env, rngs, Sp);
    ag->VT_ACCESS start_ep(ag, rngs, Sp);
    uint t = 0;
    real G = 0;

    ALLEGRO_TIMER *redraw_timer = NULL;
    if (steps_per_vis) {
        init_allegro();
        redraw_timer = al_create_timer(1.0 / FPS);
        init_die(redraw_timer, "redraw_timer")
        al_start_timer(redraw_timer);
        al_register_event_source(event_queue,
          al_get_timer_event_source(redraw_timer));
    }

    uint frame = 0;
    uint steps_till_vis = 0;
    ALLEGRO_EVENT event;
    // skip == 0: always waits for event queue
    // skip == 1: don't wait just draw, still listen in case there are events
    // skip == 2: don't listen, don't draw
    for (bool is_terminal = false; !is_terminal; frame++) {
        if (steps_per_vis && !steps_till_vis) {
            al_wait_for_event(event_queue, &event);
            switch(event.type)
            {
            case ALLEGRO_EVENT_TIMER:
                steps_till_vis = steps_per_vis;
            break;
            case ALLEGRO_EVENT_KEY_DOWN:
                if (event.keyboard.keycode == ALLEGRO_KEY_ESCAPE)
                    steps_per_vis *= 2;
            break;
            case ALLEGRO_EVENT_DISPLAY_CLOSE:
                putchar('\n');
                exit(0);
            }
        } else {
            steps_till_vis--;
            // the `draw` function
            real R;
            ag->VT_ACCESS choose_action(ag, rngs, Sp, &A, t);
            is_terminal = env->VT_ACCESS
                transition(env, rngs, Sp, &A, &R, nSp, t);
            G += R;
            ag->VT_ACCESS update(ag, rngs, Sp, &A, R, nSp, t, is_terminal);
            // uhhhhh somethings wrong
            if (t % 10000 == 9999)
                printf("uhhhh... %u\n", t);
            if (flags & RUN_GETCH) {
                printf("---- t=%u\n", t);
                printf("S:");
                print_elem(Sp, &env->SVT_ACCESS state_space);
                printf("A:");
                print_elem(&A, &env->SVT_ACCESS fixed_action_space);
                printf("R: %.4g\n", R);
                printf("S':");
                print_elem(nSp, &env->SVT_ACCESS state_space);
                getchar();
            }
            // vis
            if (!steps_till_vis)
                vis(env, ag, Sp, &A, R, nSp, t, visinfo);

            tmp = Sp;
            Sp = nSp;
            nSp = tmp;
            t++;
        }
    }

    free_elem(&A);
    free_elem(&S);
    free_elem(&nS);
    if (redraw_timer) {
        al_unregister_event_source(event_queue,
          al_get_timer_event_source(redraw_timer));
        al_destroy_timer(redraw_timer);
    }
    *outG = G;
    return t;
}

void make_mcar_experiment(Environment **env, Agent **ag, uint agi,
                          HParam *hpt) {
    MountainCar *mcar = custom_malloc(sizeof *mcar);
    make_MountainCar(mcar, MC_FL_STOPSHORT);
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

    if (agi == 0) {
        SarsaLambda *sl = custom_malloc(sizeof *sl);
        make_SarsaLambda(sl, (*env), qfn, (Policy*)epgreedy,
            hpt[1].r, hpt[2].r, hpt[3].r);
        *ag = (Agent*) sl;
    } else if (agi == 1) {
        QTNatSarsaLambda *qnsl = custom_malloc(sizeof *qnsl);
        make_QTNatSarsaLambda(qnsl, (*env), qfn, (Policy*)epgreedy,
            hpt[1].r, hpt[2].r, hpt[3].r);
        *ag = (Agent*) qnsl;
    } else {
        abort();
    }
}

void free_mcar_experiment(Environment *env_, Agent *ag_, uint agi) {
    EpGreedy *epgreedy;
    QFn *qfn;
    if (agi == 0) {
        SarsaLambda *ag = (SarsaLambda*) ag_;
        epgreedy = (EpGreedy*) ag->pi;
        qfn = (QFn*) ag->qfn;
        free_SarsaLambda(ag);
    } else if (agi == 1) {
        QTNatSarsaLambda *ag = (QTNatSarsaLambda*) ag_;
        epgreedy = (EpGreedy*) ag->pi;
        qfn = (QFn*) ag->qfn;
        free_QTNatSarsaLambda(ag);
    } else {
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

typedef struct {
    void (*make_exp)(Environment **env, Agent **ag, uint agi, HParam *hpt);
    void (*free_exp)(Environment **env, Agent **ag, uint agi, HParam *hpt);
    HParam **best_hpt_for_ag;
    uint nagents;
} Experiment;



// -------------------- Visualization --------------------------

void mcar_lin_fb_vis(Environment *env_, Agent *ag_, Elem *S, Elem *A, float R,
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

    static float val_max = 8.f, val_min = -40.f;
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



int main(int argc, char **argv) {
    // #define tmin 1000
    // #define tmax 1
    // uint c[1 + (tmax > tmin ? tmax : tmin)] = {0};
    // for (uint i = 0; i < 5000; i++) {
    //     float s = rand_exp(tmin, tmax);
    //     c[(uint)s]++;
    //     printf("%.2f ", s);
    // }
    // uint s = 0;
    // for (uint i = rmin(tmin, tmax); i < rmax(tmin, tmax)+1; i++) {
    //     s += c[i];
    //     printf("[%u]:%u\n", i, c[i]);
    // }
    // printf("s=%u\n", s);
    // return 0;

    // printf("RANDMAX=%u \nRAND_MAX/(1.f+RAND_MAX)=%.50f\n%u",
    //     RAND_MAX,
    //     RAND_MAX/(1.f+RAND_MAX),
    //     RAND_MAX/(1.f+RAND_MAX) < 1.f);
    // print_expr(RAND_MAX, "%u");
    // return 0;

    // for (uint i = 0; i < 100; i++) {
    //     float s = rand_exp(1e-5, 0.5f);
    //     printf("%.5f\n", log10f(s));
    // }
    // return 0;
    // for (uint i = 0; i < 100; i++)
    //     printf("r: %.9f\n", rand_float(s));
    // return 0;


    #define HYPERPARAM_SEARCH

    #ifdef HYPERPARAM_SEARCH

        uint nhparam_tuples = 2000;
        // freopen("./paramsearch.txt", "w", stdout);
    #else
        all_tests();
        make_MountainCar(&mcar, 0);
    #endif

    /*
    best hparams
    sarsa(lambda):
    ep=0.00005241 al=0.06805436 ga=0.99874651 la=0.38837796
    mean ret=-195.18


    qtnat sarsa(lambda)
    ep=0.00008165 al=0.07211637 ga=0.99987674 la=0.61488545
    mean ret=-167.16
    */

    // HParam best_mcar_hparam[2][4] = {
    //     {0.00005241, 0.06805436, 0.99874651, 0.38837796},
    //     {0.00008165, 0.07211637, 0.99987674, 0.61488545}
    // };

    #ifdef HYPERPARAM_SEARCH
        uint nepisodes = 20;
        uint ntrials = 20;

        HParamsTupleSearch hptss[] = {
            {
                .nhparams = 4,
                .hparam_search = (HParamSearch[]){
                    {
                        .name="ep",
                        .sampler = rand_exprange_writo,
                        .sset={.type='r', .as.range={1e-5, 0.3}}
                    },
                    {
                        .name="al",
                        .sampler = rand_exprange_writo,
                        .sset={.type='r', .as.range={1e-4, 0.1}}
                    },
                    {
                        .name="ga",
                        .sampler = rand_1m_exprange_writo,
                        .sset={.type='r', .as.range={5e-4, 0.9}}
                    },
                    {
                        .name="la",
                        .sampler = rand_1m_exprange_writo,
                        .sset={.type='r', .as.range={1e-4, 0.9}}
                    },
                }
            },
            {
                .nhparams = 4,
                .hparam_search = (HParamSearch[]){
                    {
                        .name="ep",
                        .sampler = rand_exprange_writo,
                        .sset={.type='r', .as.range={1e-5, 0.3}}
                    },
                    {
                        .name="al",
                        .sampler = rand_exprange_writo,
                        .sset={.type='r', .as.range={5e-4, 0.1}}
                    },
                    {
                        .name="ga",
                        .sampler = rand_1m_exprange_writo,
                        .sset={.type='r', .as.range={1e-4, 0.9}}
                    },
                    {
                        .name="la",
                        .sampler = rand_1m_exprange_writo,
                        .sset={.type='r', .as.range={1e-4, 0.9}}
                    },
                }
            },
        };

        for (uint agi = 0; agi < 1; agi++) {
            HParamsTupleSearch hpts = hptss[agi];

            HParam *best_hpt = custom_malloc(hpts.nhparams * sizeof(HParam));
            real best_ret_mean = FLOAT_VERY_SMALL;

#ifdef _OPENMP
            #pragma omp parallel default(firstprivate) shared(best_ret_mean)
            {
                int tid = omp_get_thread_num();
                int nthreads = omp_get_num_threads();
#else
            {

                int tid = 0;
                int nthreads = 1;
#endif
                if (tid == 0)
                    printf("nthreads=%i\n", nthreads);
                time_t time_seed = time(NULL);
                RngState rngs = {.x={
                    // idk just random linear things, doesnt matter too much
                    time_seed*995919232993 + 1893857001821,
                    (time_seed+95821)*112631198134567 + 380112,
                    (time_seed+1329761)*113333333337777 + 71982901,
                    (time_seed+17829291823)*69696969696969 + 420420420
                }};
                for (uint i = 0; i < tid; i++)
                    rand_jump(&rngs);

                HParam *hpt = custom_malloc(hpts.nhparams * sizeof(HParam));
                HParam *local_best_hpt =
                    custom_malloc(hpts.nhparams * sizeof(HParam));
                real local_best_ret_mean = FLOAT_VERY_SMALL;

                uint hpti_start = nhparam_tuples *  tid    / nthreads;
                uint hpti_end   = nhparam_tuples * (tid+1) / nthreads;
                // printf("tid %u/%u: [%u, %u)\n"
                //        "first rng: %lu\n",
                //        tid, nthreads, hpti_start, hpti_end,
                //        rand_u64(s));
                // getchar();

                // malloc & free are threadsafe since C11
                Agent *ag;
                Environment *env;
                for (uint hpti = hpti_start; hpti < hpti_end; hpti++) {
                    // draw hparam from specified distributions
                    sample_hparams_tuple(&rngs, &hpts, hpt);
                    make_mcar_experiment(&env, &ag, agi, hpt);
                    if (hpti == 0)
                        printf("Agent name: %s\n", ag->VT_ACCESS name);
                    real ret_mean = 0; // mean return across trials & eps
                    for (uint tr = 0; tr < ntrials; tr++) {
                        ag->VT_ACCESS reset(ag);
                        for (uint e = 0; e < nepisodes; e++) {
                            float G;
                            uint T = run_episode(&rngs, env, ag, &G,
                                0, NULL, &e, 0);
                            // update mean return
                            float mrlr = 1.f / (1 + tr*nepisodes + e);
                            ret_mean = ret_mean*(1-mrlr) + G*mrlr;
                            // printf("%u ", T);
                            // fflush(stdout);
                            if (T >= MC_STOPSHORT_MAX_T) {
                                goto stuck;
                            }
                        }
                    }

                    if (ret_mean > local_best_ret_mean) {
                        local_best_ret_mean = ret_mean;
                        memcpy(local_best_hpt, hpt,
                            hpts.nhparams * sizeof *hpt);
                        // printf("%u/%u: meanret=%.2f\n",
                        //        hpti, nhparam_tuples, ret_mean);
                        // print_hparams_tuple(&hpts, hpt);
                    }

                    stuck: // exit stuck episode

                    free_mcar_experiment(env, ag, agi);
                }

                #pragma omp critical
                    if (local_best_ret_mean > best_ret_mean) {
                        best_ret_mean = local_best_ret_mean;
                        memcpy(best_hpt, local_best_hpt,
                            hpts.nhparams * sizeof *hpt);
                    }

                free(hpt);
                free(local_best_hpt);
            }
            printf("Best hparam_tuple with mean ret=%.2f\n", best_ret_mean);
            print_hparams_tuple(&hpts, best_hpt);
            free(best_hpt);
        }

    #endif // HYPERPARAM_SEARCH

    #define TRIALS 1000
    #define EPISODES 20
    #define SL_VIS 0

    // real ret_mean[EPISODES] = {0}; // mean return at ep# over trials
    // for (uint agi = 0; agi < sizeof(agents)/sizeof(Agent*); agi++) {
    //     Agent *ag = agents[agi];
    //     printf("Agent name: %s\n", ag->VT_ACCESS name);
    //     for (uint tr = 0; tr < TRIALS; tr++) {
    //         ag->VT_ACCESS reset(ag);
    //         for (uint e = 0; e < EPISODES; e++) {
    //             float G;
    //             uint T = run_episode(env, ag, &G,
    //                                  SL_VIS, mcar_lin_fb_vis, &e,
    //                                  0);
    //             float mrlr = 1.f / (1 + tr);
    //             ret_mean[e] = ret_mean[e]*(1-mrlr) + G*mrlr;
    //             // printf("%u\t", T);
    //             // fflush(stdout);
    //         }
    //         // putchar('\n');
    //     }
    //     puts("Mean:");
    //     for (uint e = 0; e < EPISODES; e++) {
    //         printf("%.0f\t", ret_mean[e]);
    //     }
    //     putchar('\n');
    // }

    destroy_allegro();

    return 0;
}
