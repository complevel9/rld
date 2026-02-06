#ifndef HEADER_MOUNTAINCAR_H
#define HEADER_MOUNTAINCAR_H

#include "rl.h"

typedef struct {
    float x, vx;
} MC_State;

typedef enum {
    MC_backward = 0,
    MC_none = 1,
    MC_forward = 2
} MC_Action;

extern EnvVT MC_EnvVT;

#endif
