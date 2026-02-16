#ifndef HEADER_MOUNTAINCAR_H
#define HEADER_MOUNTAINCAR_H

#include "rl.hpp"

constexpr float MC_FORCE = 0.001f, MC_GRAV = 0.0025f,
                MC_LEFT = -1.2f, MC_RIGHT = 3.1415926535897932f/6.f,
                MC_SPEED_MAX = 0.07,
                MC_TIME_REWARD = -1.0f,
                MC_WIN_REWARD = 100.0f;

struct MC_State {
    float x, vx;
};

constexpr uint
    MC_backward = 0,
    MC_none = 1,
    MC_forward = 2;

typedef uint MC_Action;

struct MC_Env : Env<MC_State, MC_Action> {
    typedef MC_State State;
    typedef MC_Action Action;

    float best;

    void begin_episode();
    MC_State start_state();
    bool is_terminal(MC_State *s, uint t);
    bool transition(uint t, MC_State *s, MC_Action *a, real *reward, MC_State *ss);
    uint num_actions_at(MC_State *s);
};
#endif
