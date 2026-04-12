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
    char *name;
    SetSampler sampler;
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


// -------------------- Experinments ---------------------------

typedef struct {
    void (*make_exp)(Environment **env, Agent **ag, uint agi, HParam *hpt,
                     bool search);
    void (*free_exp)(Environment  *env, Agent  *ag, uint agi);
    Visfn visfn;
    HParamsTupleSearch *hptss;
    HParam **best_hpt_for_ag;
    uint nagents;
    uint nepisodes, ntrials;
    uint nsamples_hpts;
    uint nepisodes_hpts, ntrials_hpts;
    uint stuck_hptskip_times;
    uint steps_per_vis;
} Experiment;


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
            if (t % 20000 == 19999)
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


void search_hpt(Experiment *exp, RngState *base_rngs) {
#ifdef AG_START
    for (uint agi = AG_START; agi < AG_END; agi++)
#else
    for (uint agi = 0; agi < exp->nagents; agi++)
#endif
    {
        HParamsTupleSearch hpts = exp->hptss[agi];

        HParam *best_hpt = exp->best_hpt_for_ag[agi];
        real best_ret_mean = FLOAT_VERY_SMALL;

        // note that inside omp parallel, OpenBLAS always use 1 thread
        #pragma omp parallel default(firstprivate) shared(best_ret_mean)
        {
#ifdef _OPENMP
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
#else
            int tid = 0;
            int nthreads = 1;
#endif
            if (tid == 0)
                printf("nthreads=%i\n", nthreads);
            // setup separate rng for each thread, copy from base_rngs
            RngState rngs = *base_rngs;
            for (uint i = 0; i < tid; i++)
                rand_jump(&rngs);

            // alloc mem for current hpt being looked at and the best one so
            // far (of this thread)
            HParam *hpt = custom_malloc(hpts.nhparams * sizeof(HParam));
            HParam *local_best_hpt =
                custom_malloc(hpts.nhparams * sizeof(HParam));
            real local_best_ret_mean = FLOAT_VERY_SMALL;

            // divide workload for threads, only need to approximately sum to
            // exp->nsamples_hpts.
            uint nsamples_per_thread = (nthreads + exp->nsamples_hpts) / nthreads;

            Agent *ag;
            Environment *env;
            for (uint hpti = 0; hpti < nsamples_per_thread; hpti++) {
                // sample hparam from specified distributions
                sample_hparams_tuple(&rngs, &hpts, hpt);
                exp->make_exp(&env, &ag, agi, hpt, true);
                if (hpti == 0 && tid == 0)
                    printf("Agent name: %s\n", ag->VT_ACCESS name);
                real ret_mean = 0; // mean return across trials & eps
                uint total_eps = 0;
                for (uint tr = 0; tr < exp->ntrials_hpts; tr++) {
                    ag->VT_ACCESS reset(ag);
                    for (uint e = 0; e < exp->nepisodes_hpts; e++) {
                        float G;
                        uint T = run_episode(&rngs, env, ag, &G,
                            0, NULL, &e, 0);
                        total_eps++;
                        // update mean return
                        float mrlr = 1.f / total_eps;
                        ret_mean = ret_mean*(1-mrlr) + G*mrlr;
                        if (T >= exp->stuck_hptskip_times)
                            goto stuck;
                    }
                }
                // update thread's best
                if (ret_mean > local_best_ret_mean) {
                    local_best_ret_mean = ret_mean;
                    memcpy(local_best_hpt, hpt,
                        hpts.nhparams * sizeof *hpt);
                }

                stuck: // exit stuck episode and skips the tuple

                exp->free_exp(env, ag, agi);
            }
            // reduce max from threads
            #pragma omp critical
                if (local_best_ret_mean > best_ret_mean) {
                    best_ret_mean = local_best_ret_mean;
                    memcpy(best_hpt, local_best_hpt,
                        hpts.nhparams * sizeof *hpt);
                }

            free(hpt);
            free(local_best_hpt);

            // copy from thread that did the most jump to avoid the used
            // "dirty" range. this is just extra precaution.
            if (tid == nthreads - 1)
                *base_rngs = rngs;
        }
        printf("Best hparam_tuple with mean ret=%.2f\n", best_ret_mean);
        print_hparams_tuple(&hpts, best_hpt);
    }
}

void run_exp(Experiment *exp, RngState *rngs) {
    real *ret_sum = calloc(exp->nepisodes, sizeof(real)); // mean return at ep# over trials
#ifdef AG_START
    for (uint agi = AG_START; agi < AG_END; agi++)
#else
    for (uint agi = 0; agi < exp->nagents; agi++)
#endif
    {
        Agent *ag;
        Environment *env;
        exp->make_exp(&env, &ag, agi, exp->best_hpt_for_ag[agi], false);
        printf("Agent name: %s\n", ag->VT_ACCESS name);
        for (uint tr = 0; tr < exp->ntrials; tr++) {
            ag->VT_ACCESS reset(ag);
            for (uint ep = 0; ep < exp->nepisodes; ep++) {
                float G;
                uint T = run_episode(rngs, env, ag, &G,
                                     exp->steps_per_vis, exp->visfn, &ep,
                                     0);
                ret_sum[ep] += G;
                printf("%u\t", T);
                fflush(stdout);
                (void)T;
            }
            putchar('\n');
        }
        exp->free_exp(env, ag, agi);
        puts("Mean:");
        for (uint ep = 0; ep < exp->nepisodes; ep++)
            printf("%.0f\t", ret_sum[ep] / exp->ntrials);
        putchar('\n');
    }
    free(ret_sum);
}
