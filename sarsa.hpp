#ifndef HEADER_SARSA_H_
#define HEADER_SARSA_H_

#include "rl.hpp"

SAQ_TEMPL struct SarsaLambda_Algo : Algo<State, Action, QFnI> {
    float step_size;
    float trace_decay;
    void *value_est;
    void begin_episode(QFnI *Q) {

    }

    Action choose_action(State *s) {

    }

    void update(QFnI *Q, uint t, State *s, Action *a, real r, State *ss,
                bool is_terminal)
    {

    };
};


#endif
