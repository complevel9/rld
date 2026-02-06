#include "mountaincar.h"

#include <math.h>

// implementation same as gymnasium's mountain car

static MC_State MC_sample_start(void *env)
{
    return (MC_State){.x = 0.0f, .vx = 0.0f};
}

static bool MC_transition(void *env, MC_State s, MC_Action a, real *r, MC_State *ss)
{
    ss->vx += (a - 1)*0.001f - 0.0025f*cosf(3.0f*s.x);
    ss->x = s.x + ss->vx;
    if (ss->x < -1.2f)
    {
        ss->x = -1.2f;
        if (ss->vx < 0.f)
            ss->vx = 0.f;
    }
    if (ss->x >= M_PI/6.f)
    {
        ss->x = M_PI/6.f;
        ss->vx = 0.f;
        *r = 100.0f;
        return true;
    }
    *r = 0.0f;
    return false;
}

// gcc will complain when type punning structs like State,
// this though, looks more clean
// (will probably crash with incorrect signature anyways)
EnvVT MC_EnvVT = {
    .sample_start = (void*)&MC_sample_start,
    .transition   = (void*)&MC_transition
};
