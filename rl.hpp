#ifndef HEADER_RL_H_
#define HEADER_RL_H_

#include <stdbool.h>
#include "commontypes.h"

typedef unsigned int uint;

// you get 16 bytes to encode whatever state or action, use the union
// or typecast them yourself
// if that's too small, allocate the memory yourself and use the void *p
// remember to free it afterwards though
typedef union Generic16 {
    char str[16];
    int i;
    uint ui;
    float f;
    double d;
    void *p;
} State, Action;

//typedef struct {
//
//} StateVT, ActionVT;

// environment functions: sample start state, sample transition (also checks
// if terminal transition)
typedef struct EnvVT {
    void  (*begin_episode)(void *env);
    State (*sample_start) (void *env);
//  bool  (*is_terminal)  (void *env, State s, uint t);
    bool  (*transition)   (void *env, State s, Action a, real *reward,
                           State *ss);
} EnvVT;

// agent functions: choose action, update
// update is responsible for freeing a state or action
typedef struct AgentVT {
    void   (*begin_episode)(void *ag);
    Action (*choose_action)(void *ag, State s);
    void   (*update)       (void *ag, uint t, State s, Action a, real r,
                            State ss, bool is_terminal);
} AgentVT;

// experiemnt setup
typedef struct {
    AgentVT agv;
    EnvVT envv;
//    StateVT statev;
//    ActionVT actionv;
    void *ag, *env;
} ExperimentSetup;



void run_episode(ExperimentSetup es);

#endif
