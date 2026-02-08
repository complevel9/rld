#include <iostream>
#include <cstdlib>
#include <vector>

#include "nd.hpp"
#include "rl.hpp"
#include "mountaincar.hpp"
#include "valuefn.hpp"
#include "sarsa.hpp"

/*

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
*/

#define szp(x) \
printf("sizeof " #x " %lu\n", sizeof(x))

using namespace std;

SA_TEMPL struct Empty_QFn : QFn<State,Action> {
    real Q_theta(State *s, Action *a) {return 10;}
    void dQ_addto(State *s, Action *a, Vec<float> *z) {}
};

SA_TEMPL struct Stupid_Algo : Algo<State,Action,Empty_QFn<State,Action>> {
    uint t;
    void begin_episode(Empty_QFn<State,Action> *Q) { t = 0; }
    Action choose_action(State *s) {
        t++; if (t <= 29) return MC_forward; else if (t <= 60 )return MC_backward; return MC_forward;}
    void update(Empty_QFn<State,Action> *Q, uint t, State *s, Action *a, real r, State *ss,
                bool is_terminal) {};
};

int main(int argc, char **argv)
{
    std::vector<uint> d = {100,20,30};
    NdArray<float> a(d.size(), &d[0]);
    for (uint i = 0; i < a.vol; i++)
        a.data[i] = i;
    a.print_dim();
    std::cout << a[&(std::vector<uint>({10, 5, 7})[0])] << '\n';
    std::cout << a[&(std::vector<uint>({10, 5, 8})[0])] << '\n';
    std::cout << &a[&(std::vector<uint>({5, 6, 8})[0])] - &a[&(std::vector<uint>({5, 6, 7})[0])] << '\n';

    szp(NdArray<float>);

    MC_Env mcenv;
    Empty_QFn<MC_State, MC_Action> Q;
    Stupid_Algo<MC_State, MC_Action> algo;
    run_episode<MC_State, MC_Action>(mcenv, algo, Q);
    /*
    szp(State);
    szp(Action);
    szp(AgentVT);
    szp(EnvVT);
    szp(ExperimentSetup);
    szp(size_t);
    szp(long int);
    szp(long long int);

    experiment = env + valuefn + algo
    agent = algo + valuefn
    valuefn = parameter + Q + dQ + state action mapping
    ActionValueFn AV = {theta, Q, dQ, mapping from };
    Sarsa = create agent(sarsa params, Sarsa_AgentVT, AV)

    create state aggregation valuefn
    how many dim is state?
    how many dim is action?
    both tied to environment, but maybe flattenable
    how to describe state/action space: cartesian product
    but dont. just create a custom function approx for mountain car, cartpole etc
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
    */
}
