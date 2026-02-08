#ifndef HEADER_RL_H_
#define HEADER_RL_H_

#include "commontypes.hpp"
#include "nd.hpp"

#define SA_TEMPL template<typename State, typename Action>

SA_TEMPL struct Env {
    virtual void begin_episode() = 0;
    virtual State start_state() = 0;
    virtual bool is_terminal(State *s, uint t) = 0;
    virtual bool transition(uint t, State *s, Action *a, real *reward,
                            State *ss) = 0;
};

//template<typename State> struct VFn {};

// Action value smooth parametric approximation
SA_TEMPL struct QFn {
    Vec<float> theta;
    // Q_\theta(s, a)
    virtual real Q_theta(State *s, Action *a) = 0;
    // z += \partial Q_\theta(s, a) / partial \theta
    virtual void dQ_addto(State *s, Action *a, Vec<float> *z) = 0;
};

#define SAQ_TEMPL template<typename State, typename Action, typename QFnI>

SAQ_TEMPL struct Algo {
    virtual void begin_episode(QFnI *Q) = 0;
    virtual Action choose_action(State *s) = 0;
    virtual void update(QFnI *Q, uint t, State *s, Action *a, real r,
                        State *ss, bool is_terminal) = 0;
};

template<typename State, typename Action,
         typename EnvI, typename QFnI, typename AlgoI>
void run_episode(EnvI env, AlgoI algo,
                          QFnI Q)
{
    bool is_terminal = false;
    State s = env.start_state();
    env.begin_episode();
    algo.begin_episode(&Q);

    for (uint t = 0;; t++)
    {
        Action a = algo.choose_action(&s);
        real r;
        State ss;
        is_terminal = env.transition(t, &s, &a, &r, &ss);
        algo.update(&Q, t, &s, &a, r, &ss, is_terminal);
        if (is_terminal) break;
        s = ss;
    }
}

#endif
