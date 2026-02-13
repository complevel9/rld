#include <iostream>
#include <cstdlib>
#include <vector>

#include "nd.hpp"
#include "rl.hpp"
#include "mountaincar.hpp"
#include "valuefn.hpp"
#include "sarsa.hpp"

#define szp(x) \
printf("sizeof " #x " %lu\n", sizeof(x))

using namespace std;

uint range_splitter_clamp(float x, float a, float b, uint num_splits)
{
    return (uint)((clamp(x, a, b) - a) / (b - a) * num_splits);
}

uint range_splitter_zero(float x, float a, float b, uint num_splits)
{
    if ((x < a) | (x > b))
        return num_splits; // invalid x so return invalid index
    return (uint)((x - a) / (b - a) * num_splits);
}

struct MCAggregateQFn : QFn<MC_Env> {
    uint x_splits, vx_splits;
    NdArray<float> theta_as_nd;
    MCAggregateQFn(uint x_splits_, uint vx_splits_)
        : x_splits(x_splits_), vx_splits(vx_splits_),
        theta_as_nd(&theta, 3, x_splits_, vx_splits_, 3)
    { }

    uint SA_to_flat(MC_State *s, MC_Action *a)
    {
        return theta_as_nd.flatpos(
            range_splitter_clamp(s->x, MC_LEFT, MC_RIGHT, x_splits),
            range_splitter_clamp(s->vx, -MC_SPEED_MAX, MC_SPEED_MAX,
                                 vx_splits),
        *a);
    }

    real Q_theta(MC_State *s, MC_Action *a)
    {
        return theta[SA_to_flat(s, a)];
    }

    void dQ(MC_State *s, MC_Action *a, Vec<float> *g)
    {
        g->set_all(0.0f);
        (*g)[SA_to_flat(s, a)] = 1.0f;
    }

    void dQ_addto(MC_State *s, MC_Action *a, Vec<float> *z)
    {
        (*z)[SA_to_flat(s, a)] += 1.0f;
    }
};
//struct tt {int a;};
struct MCDirectFeatureLinearQFn : QFn<MC_Env> {
    MCDirectFeatureLinearQFn() : QFn(4)
    {}

    real Q_theta(MC_State *s, MC_Action *a)
    {
        return s->x*theta[0] + s->vx*theta[1] + ((int)(*a) - 1)*theta[2];
    }

    void dQ(MC_State *s, MC_Action *a, Vec<float> *g)
    {
        (*g)[0] = s->x;
        (*g)[1] = s->vx;
        (*g)[2] = ((int)*a) - 1;
    }

    void dQ_addto(MC_State *s, MC_Action *a, Vec<float> *z)
    {
        (*z)[0] += s->x;
        (*z)[1] += s->vx;
        (*z)[2] += ((int)*a) - 1;
    }
};

template<class EnvI, class QFnI> struct Stupid_Algo : Algo<EnvI,QFnI> {
    typedef typename EnvI::State State;
    typedef typename EnvI::Action Action;

    uint t;
    void begin_episode(QFnI *Q) { t = 0; }
    Action choose_action(State *s) {
        t++; if (t <= 29) return MC_forward; else if (t <= 60 )return MC_backward; return MC_forward;}
    void update(QFnI *Q, uint t, State *s, Action *a, real r, State *ss,
                bool is_terminal) {};
};

int main(int argc, char **argv)
{
    std::vector<uint> d = {100,20,30};
    NdArray<float> a(nullptr, d.size(), &d[0]);
    for (uint i = 0; i < a.size; i++)
        a.data[i] = i;
    a.print_dim();
    std::cout << a[&(std::vector<uint>({10, 5, 7})[0])] << '\n';
    std::cout << a[&(std::vector<uint>({10, 5, 8})[0])] << '\n';
    std::cout << &a[&(std::vector<uint>({5, 6, 8})[0])] - &a[&(std::vector<uint>({5, 6, 7})[0])] << '\n';

    szp(NdArray<float>);

    MC_Env mcenv;
    MCAggregateQFn Q(8, 8);
    //MCDirectFeatureLinearQFn Q;
    EpsilonGreedy<MC_Env, MCAggregateQFn> epgreedy;
    epgreedy.Q = &Q;
    epgreedy.epsilon = 0.02;
    epgreedy.env = &mcenv;
    epgreedy.random_tiebreak = true;
    SarsaLambda_Algo<MC_Env, MCAggregateQFn, EpsilonGreedy<MC_Env,MCAggregateQFn>>        sarsa_lambda(0.05, 0.95, 0.5, &Q, &epgreedy);
    NatSarsaLambda_Algo<MC_Env, MCAggregateQFn, EpsilonGreedy<MC_Env,MCAggregateQFn>> nat_sarsa_lambda(0.05, 0.95, 0.5, &Q, &epgreedy);

    printf("Q.xs %u; Q.vxs %u\n", Q.x_splits, Q.vx_splits);
    Q.theta_as_nd.print_dim();
    srand(0);
    Q.theta.set_all(0.0f); // optimistic initialisation
    for (uint ep = 0; ep < 20; ep++)
    {
        uint steps = run_episode<MC_Env, MCAggregateQFn>(&mcenv, &sarsa_lambda, false);
        printf("SL:  ep %u took %u steps\n", ep, steps);
    }
    srand(0);
    Q.theta.set_all(0.0f); // optimistic initialisation
    for (uint ep = 0; ep < 20; ep++)
    {
        //epgreedy.epsilon = 1.0f / (2+ep);
        uint steps = run_episode<MC_Env, MCAggregateQFn>(&mcenv, &nat_sarsa_lambda, false);
        printf("NSL: ep %u took %u steps\n", ep, steps);
//        for (uint j = 0; j < Q.vx_splits; j++)
//        {
//            for (uint i = 0; i < Q.x_splits; i++)
//            {
//                uint k = i*Q.vx_splits*3 + j*3;
//                printf("%7.1f|%7.1f|%7.1f", Q.theta[k],Q.theta[k+1],Q.theta[k+2]);
//                if (i < 3) printf("\t");
//            }
//            printf("\n");
//
//        }
    }

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
