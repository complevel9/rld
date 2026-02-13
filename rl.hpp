#ifndef HEADER_RL_HPP_
#define HEADER_RL_HPP_

#include "common.hpp"
#include "nd.hpp"

#define SA_TEMPL template<class State, class Action>

SA_TEMPL struct Env {
    virtual void begin_episode() = 0;
    virtual State start_state() = 0;
    virtual bool is_terminal(State *s, uint t) = 0;
    virtual bool transition(uint t, State *s, Action *a, real *reward,
                            State *ss) = 0;
};

//template<class State> struct VFn {};

// Action value smooth parametric approximation
template<class EnvI> struct QFn {
    typedef typename EnvI::State State;
    typedef typename EnvI::Action Action;
    Vec<float> theta;
    QFn(uint vecsize) : theta(vecsize) {};
    QFn() {};
    // Q_\theta(s, a)
    virtual real Q_theta(State *s, Action *a) = 0;
    // z += \partial Q_\theta(s, a) / partial \theta
    virtual void dQ      (State *s, Action *a, Vec<float> *g) = 0;
    virtual void dQ_addto(State *s, Action *a, Vec<float> *z) = 0; // make optional
};

#define SAQ_TEMPL template<class State, class Action, class QFnI>

template<class EnvI, class QFnI> struct Algo {
    typedef typename EnvI::State State;
    typedef typename EnvI::Action Action;

    virtual void begin_episode(State *s) = 0;
    virtual Action choose_action(State *s) = 0;
    virtual void update(uint t, State *s, Action *a, real r,
                        State *ss, bool is_terminal) = 0;
};

template<class EnvI, class QFnI, class AlgoI>
uint run_episode(EnvI* env, AlgoI* algo, bool b)
{
    typedef typename EnvI::State State;
    typedef typename EnvI::Action Action;
    uint t;
    bool is_terminal = false;
    env->begin_episode();
    State s = env->start_state();
    algo->begin_episode(&s);

    for (t = 0;; t++)
    {
        Action a = algo->choose_action(&s);
        real r;
        State ss;
        is_terminal = env->transition(t, &s, &a, &r, &ss);
        algo->update(t, &s, &a, r, &ss, is_terminal);
        if (b && t > 0 && t % 20 == 0) {
            printf("t = %u\n", t);
            QFnI *Q = algo->Q;
            for (uint j = 0; j < Q->vx_splits; j++)
            {
                for (uint i = 0; i < Q->x_splits; i++)
                {
                    uint k = i*Q->vx_splits*3 + j*3;
                    printf("%7.1f|%7.1f|%7.1f", Q->theta[k],Q->theta[k+1],Q->theta[k+2]);
                    if (i < 3) printf("\t");
                }
                printf("\n");
            }
            getchar();
        }
        if (is_terminal) break;
        s = ss;
    }
    return t;
}

#endif
