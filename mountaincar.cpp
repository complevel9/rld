#include "mountaincar.hpp"

#include <cmath>
#include <stdlib.h>

// implementation same as gymnasium's mountain car

void MC_Env::begin_episode()
{
}

MC_State MC_Env::start_state()
{
    MC_State mc = {.x = -0.5f, .vx = 0.0f};
    return mc;
}

bool MC_Env::transition(uint t, MC_State *s, MC_Action *a, real *r, MC_State *ss)
{
    ss->vx += (*a - 1)*0.001f - 0.0025f*cosf(3.0f*s->x);
    ss->x = s->x + ss->vx;
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
    printf("t=%u:\tx=%.2f\tvx=%.2f", t, ss->x, ss->vx);
    getchar();
    return false;
}

bool MC_Env::is_terminal(MC_State *s, uint t)
{
    return s->x >= M_PI/6.f;
}
