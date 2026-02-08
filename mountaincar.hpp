#ifndef HEADER_MOUNTAINCAR_H
#define HEADER_MOUNTAINCAR_H

#include "rl.hpp"

struct MC_State {
    float x, vx;
};

enum MC_Action {
    MC_backward = 0,
    MC_none = 1,
    MC_forward = 2
};

struct MC_Env : Env<MC_State, MC_Action> {
    void begin_episode();
    MC_State start_state();
    bool is_terminal(MC_State *s, uint t);
    bool transition(uint t, MC_State *s, MC_Action *a, real *reward, MC_State *ss);
};
#endif
