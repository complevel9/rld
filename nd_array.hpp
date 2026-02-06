#ifndef HEADER_ND_ARRAY_H_
#define HEADER_ND_ARRAY_H_

// see nd_array_x.h for actual code
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "commontypes.h"

// I LOVE MACROSLOP

#define dtype float
#include "nd_array_x.h"
#undef dtype

#define dtype double
#include "nd_array_x.h"
#undef dtype

#define dtype int
#include "nd_array_x.h"
#undef dtype

#define ND_FREE(f) {free((f).data);free((f).dim);}

uint nd_pos_fromdim(uint ndim, uint *dim, ...);
#define ND_POS(f, ...) nd_pos_fromdim((f).ndim, (f).dim, __VA_ARGS__)
// error "f is a pointer..." means you passed the wrong thing into the macro
#define ND_AT(f, ...) (f).data[ND_POS((f), __VA_ARGS__)]

#endif
