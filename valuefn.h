#ifndef HEADER_VALUEFN_H_
#define HEADER_VALUEFN_H_

#include "rl.h"

typedef struct {
    void *(*create_param) ();
    void  (*destroy_param)(void *q);
    real  (*Q)            (void *theta, State s, Action a);
    void  (*dQ)           (void *dest, void *theta, State s, Action a);
} QVT;

#endif
