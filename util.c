// -------------------- OOP macros ----------------------------

// define VT_COPY so that structs contains a copy of vtables, instead of a
// pointer to it. saves some extra dereferencing. the structs are small
// anyways.
//#define VT_COPY

#ifdef VT_COPY
    #define VT_PTR_ASTER
    #define VT_PTR_ARROW .
    #define VT_PTR_AMPER
    #define VT_PTR_SW(a,b) b
    #define VT_COPY_ASTER *
    #define VT_COPY_ARROW ->
    #define VT_COPY_AMPER &
    #define VT_COPY_SW(a,b) a
#else
    #define VT_PTR_ASTER *
    #define VT_PTR_ARROW ->
    #define VT_PTR_AMPER &
    #define VT_PTR_SW(a,b) a
    #define VT_COPY_ASTER
    #define VT_COPY_ARROW .
    #define VT_COPY_AMPER
    #define VT_COPY_SW(a,b) b
#endif

#define SVT_ACCESS vt VT_PTR_ARROW svt VT_COPY_ARROW
#define VT_ACCESS vt VT_PTR_ARROW

// -------------------- Common types --------------------------

typedef uint32_t uint;
typedef uint16_t ushort;
typedef uint8_t uchar;
#if 1
    typedef float real;
    #define REAL_SINGLE 1
#else
    typedef float real;
    #define REAL_SINGLE 0
#endif

// -------------------- RNG -----------------------------------

// having any custom, inlinable rng is way faster than going to glibc
// just due to cache thrashing in parallel code alone
// xorshiro256++ from https://prng.di.unimi.it/xoshiro256plusplus.c
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

typedef struct { uint64_t x[4]; } RngState;

uint64_t rand_u64(RngState *s) {
    const uint64_t result = rotl(s->x[0] + s->x[3], 23) + s->x[0];
    const uint64_t t = s->x[1] << 17;
    s->x[2] ^= s->x[0];
    s->x[3] ^= s->x[1];
    s->x[1] ^= s->x[2];
    s->x[0] ^= s->x[3];
    s->x[2] ^= t;
    s->x[3] = rotl(s->x[3], 45);
    return result;
}

real rand_real(RngState *s) {
    return (rand_u64(s) >> 10) / (real)((uint64_t)1 << 54);
}

// equivalent to 2^128 calls to next()
void rand_jump(RngState *s) {
    static const uint64_t JUMP[] = {
        0x180ec6d33cfd0aba, 0xd5a61266f0c9392c,
        0xa9582618e03fc9aa, 0x39abdc4529b1661c
    };
    uint64_t s0 = 0;
    uint64_t s1 = 0;
    uint64_t s2 = 0;
    uint64_t s3 = 0;
    for(int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
        for(int b = 0; b < 64; b++) {
            if (JUMP[i] & UINT64_C(1) << b) {
                s0 ^= s->x[0];
                s1 ^= s->x[1];
                s2 ^= s->x[2];
                s3 ^= s->x[3];
            }
            rand_u64(s);
        }
    s->x[0] = s0;
    s->x[1] = s1;
    s->x[2] = s2;
    s->x[3] = s3;
}


#ifndef M_PI
    #define M_PI 3.14159265358979323846264338328
#endif

// INFINITY is slow to process
#define FLOAT_VERY_LARGE +9.9999e+60
#define FLOAT_VERY_SMALL -9.9999e+60

// -------------------- Math utils ----------------------------

extern inline real rmin(real a, real b){return a<b ? a : b;}
extern inline real rmax(real a, real b){return a>b ? a : b;}
extern inline uint umin(uint a, uint b){return a<b ? a : b;}
extern inline uint umax(uint a, uint b){return a>b ? a : b;}
extern inline real rclamp(real v, real a, real b){return v<a ? a : v>b?b:v;}
extern inline real uclamp(uint v, uint a, uint b){return v<a ? a : v>b?b:v;}
extern inline bool req(real a, real b, real ep){return (a-b<=ep)&(b-a<=ep);}
extern inline bool req_arr(real *a, real *b, uint n, real ep) {
    for (uint i = 0; i < n; i++)
        if (!req(a[i], b[i], ep)) {
            // printf("a[%u]=%.9g b[%u]=%.9g\n", i, a[i], i, b[i]);
            return false;
        }
    return true;
}

// don't use this it's slower than libm's cosf()
#if 0
    static inline
    float cosf_x87(float x) {
        register double result;
        __asm__ __volatile__ inline (
            "fcos"
            : "=t" (result)
            : "0" (x)
        );
        return result;
    }
    static inline
    double cos_x87(double x) {
        register double result;
        __asm__ __volatile__ inline (
            "fcos"
            : "=t" (result)
            : "0" (x)
        );
        return result;
    }
    #define cosf cosf_x87
    #define cos  cos_x87
#endif


// -------------------- Distributions -------------------------

extern inline bool rand_bernoulli(RngState *s, float pr) {
    if (pr <= 0.f) return 0;
    if (pr >= 1.f) return 1;
    return rand_real(s) < pr;
    // return rand_float(s) < pr;
}
extern inline real rand_uniform(RngState *s, real a, real b) {
    return a + (b-a)*rand_real(s);
}
extern inline real rand_exprange(RngState *s, real a, real b) {
    real la = log2f(a), lb = log2f(b);
    real lr = rand_uniform(s, la, lb);
    return rclamp(b, a, exp2f(lr));
}

// -------------------- Debug utils ---------------------------

#define print_sizeof(x) printf("sizeof " #x " = %lu\n", sizeof(x))
#define print_expr(x,fmt) printf( #x " = " fmt "\n", (x))
#define sigtrap() asm volatile ("int $3")// int 0x03

// -------------------- Miscellaneous -------------------------

// somethings wrong if more than this much mem is being allocated
#define MALLOC_MAX (1024*40)

// since malloc with size 0 doesnt have portable behavior
void *custom_malloc(size_t s) {
    if (s) {
        assert(s <= MALLOC_MAX);
        void *r = malloc(s);
        assert(r);
        return r;
    }
    return 0;
}
