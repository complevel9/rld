#ifndef HEADER_VALUEFN_H_
#define HEADER_VALUEFN_H_

#include "nd.hpp"
#include "rl.hpp"




/*
its been three fucking days i still dont know how to represent this
ok. approach one.

both state and action are of type Element with some parent CartesianProduct
realistically it would just be a bunch of ints and floats in a list
then there will be a feature mapping from Element to....?
then the element can maybe be automatically converted to flat index of some
n-d array

approach two
state and action can be whatever the fuck
then some feature mapping to generic float vector, maybe sparse representation
may be not
float vector must have: addition, scale, fma

i just like the 2nd option way better, since just the automatic conversion of
flat index isnt much
*/

/*
typedef struct {
    void *(*create_param) ();
    void  (*destroy_param)(void *q);
    real  (*Q)            (void *theta, State s, Action a);
    void  (*dQ)           (void *dest, void *theta, State s, Action a);
} QVT;
*/
#endif
