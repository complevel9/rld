#include "mountaincar.hpp"

#include <math.h>
#include <stdlib.h>

// implementation same as gymnasium's mountain car

void MC_Env::begin_episode()
{
    best = -100.0f;
}

MC_State MC_Env::start_state()
{
    MC_State mc = {.x = -0.5f, .vx = 0.0f};
    return mc;
}

bool MC_Env::transition(uint t, MC_State *s, MC_Action *a, real *r, MC_State *ss)
{
    ss->vx = s->vx + (((int)*a) - 1)*MC_FORCE - MC_GRAV*cosf(3.0f*s->x);
    ss->x = s->x + ss->vx;
    if (ss->x < MC_LEFT)
    {
        ss->x = MC_LEFT;
        if (ss->vx < 0.f)
            ss->vx = 0.f;
    }
    if (ss->x >= MC_RIGHT)
    {
        ss->x = MC_RIGHT;
        ss->vx = 0.f;
        *r = MC_WIN_REWARD;
        return true;
    }
    best = std::max(best, ss->x);
    *r = MC_TIME_REWARD;

    extern bool logmc;
    if (logmc) {
        extern bool greedy;
        printf("%c t=%u: a=%4u  x=%5.2f  vx=%5.2f  best=%5.2f  \n", greedy ? '*' : ' ', t, *a, ss->x, ss->vx, best);
        greedy = false;
//        if (t % 20 == 0)
//            getchar();
    }
    return false;
}

bool MC_Env::is_terminal(MC_State *s, uint t)
{
    return s->x >= M_PI/6.f;
}

uint MC_Env::num_actions_at(MC_State *s)
{
    return 3;
}
