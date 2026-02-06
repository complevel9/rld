#include <stdio.h>
#include <stdlib.h>

#include "nd_array.h"
#include "rl.h"
#include "mountaincar.h"
#include "valuefn.h"

typedef struct {

} Aggregate;

typedef struct {
    nd_float z;
    nd_float theta;
    Action action;
    QVT qv;
} Agent;

void ag_begin_episode(void *ag_)
{
    Agent ag = *(Agent*)ag_;
    ag.z = 0;
    // ag.action = choose action
}

Action ag_choose_action(void *ag_, State s)
{
    Agent ag = *(Agent*)ag_;
    return ag.action;
}

void sarsa_update(void *ag_, uint t, State s, Action a, real r, State ss, bool is_terminal)
{
    Agent ag = *(Agent*)ag_;
    float delta = R - ag.QVT.Q(ag.theta, s, a);
    // ag.z += ag.QVT.dQ(ag.theta, s, a);
    if (is_terminal) {
        // w += alpha * delta * z
        return;
    }
    // ag.action = choose action
    delta += ag.gamma*ag.QVT.Q(ag.theta, ss, ag.action);
    // ag.z.mul(gamma * lambda);
}

int main(int argc, char **argv)
{
    #define szp(x) \
    printf("sizeof " #x " %lu\n", sizeof(x))
    szp(State);
    szp(Action);
    szp(AgentVT);
    szp(EnvVT);
    szp(ExperimentSetup);
    szp(size_t);
    szp(long int);
    szp(long long int);

    /*
    experiment = algo +
    agent = algo + valuefn
    valuefn = parameter + Q fn
    ActionValueFn AV = {theta, Q, dQ, mapping from };
    Sarsa = create agent(sarsa params, Sarsa_AgentVT, AV)

    create state aggregation valuefn
    how many dim is state?
    how many dim is action?
    both tied to environment, but maybe flattenable
    how to describe state/action space: cartesian product
    but dont. just create a custom function approx for mountain car, cartpole etc
    */
    ExperimentSetup es;
    Agent ag = {0};
    es.ag = &ag;
    es.agv = (AgentVT){
        .begin_episode = ag_begin_episode,
        .update = sarsa_update,
        .choose_action = max_choose_action,
    };
    es.env = 0;
    es.envv = MC_EnvVT;
    run_episode(es);
    return 0;
}
