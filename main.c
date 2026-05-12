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

// agent index list: select a list of agents instead of running all of them
// terminate the list with 0
#define AGI_LIST {3, 0}

#include "colormap.c"

FILE *fp_ret = NULL, *fp_theta = NULL, *fp_metric;
#include "experiment.c"
#include "exp_flip.c"
#include "exp_mcar.c"

int main(int argc, char **argv) {
    #ifdef _OPENMP
        puts("OpenMP is enabled");
    #else
        puts("OpenMP is not enabled");
    #endif

    #if HAS_CBLAS
        printf("OpenBLAS threading model: %s\n", openblas_parallel_str());
    #else
        puts("OpenBLAS is not enabled");
    #endif

    #ifdef NDEBUG
        puts("assert macros disabled");
    #else
        puts("assert macros active");
    #endif

    // all_tests();

    time_t timeseed = time(NULL);
    // time_t timeseed = 1234;
    printf("Time seed is %ld\n", timeseed);
    RngState rngs = {.x={
        // idk just random looking linear things, doesnt matter too much
        timeseed*995919232993llu + 1893857001821llu,
        (timeseed+95821llu)*112631198134567llu + 380112llu,
        (timeseed+1329761llu)*113333333337777llu + 71982901llu,
        (timeseed+17829291823llu)*69696969696969llu + 420420420llu
    }};

    #if 0
        search_hpt(&mcar_exp, &rngs);
    #endif // HYPERPARAM_SEARCH

    // run_exp(&mcar_exp, &rngs);

    FILE* fp = NULL;
    // fp = stdout;
    // fp = fopen("flip.csv", "w");
    fp_ret    = fopen("ret.csv",   "w");
    fp_theta  = fopen("theta.csv", "w");
    fp_metric = fopen("metric.csv", "w");

    run_exp(&flip_exp, &rngs, fp);

    // fclose(fp);
    fclose(fp_ret);
    fclose(fp_theta);
    fclose(fp_metric);

    destroy_allegro();

    return 0;
}
