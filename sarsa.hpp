#ifndef HEADER_SARSA_H_
#define HEADER_SARSA_H_

#include "rl.h"

typedef struct {
    float step_size;
    float trace_decay;
    void *value_est;

} Algo_Sarsa;


#endif
