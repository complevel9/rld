#include "rl.h"

void run_episode(ExperimentSetup es)
{
    bool is_terminal = false;
    State s = es.envv.sample_start(es.env);
    es.envv.begin_episode(es.env);
    es.agv.begin_episode(es.ag);

    for (uint t = 0;; t++)
    {
        Action a = es.agv.choose_action(es.ag, s);
        real r;
        State ss;
        is_terminal = es.envv.transition(es.env, s, a, &r, &ss);
        es.agv.update(es.ag, t, s, a, r, ss, is_terminal);
        if (is_terminal) break;
        s = ss;
    }
}
