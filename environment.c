// -------------------- Environments --------------------------

typedef struct {
    Space state_space;
    Space fixed_action_space;
    char *name;
    bool action_space_is_fixed;
    bool action_space_is_discrete;
    bool deterministic_transition;
} EnvironmentSVT; // "static virtual table"

typedef struct {
    // lol you cant do this in C++?? why no virtual static member var
    // accessible from obj. iso wg pls fix
    // you can literally get class name at run time too. without having to use
    // a getter func. its so convenient for a specific niche. zero overhead.
    // just a bit weird here since i can switch to copying the whole vtable,
    // which is not C++ behaviour, then these cant be copied as normal var
    // but need to be ptr to still be shared: there must be at least 1
    // dereferencing for the sharing. making it always a pointer solves the
    // issue, but now if objects only have vtable ptr then there's 2 derefs
    // instead of one. at the end of the day, vtables are just constant tables
    // shared within a class. usually they only have constant fn ptrs in them,
    // but why not just put other things in it as well. they can be non const
    // too.
    EnvironmentSVT VT_COPY_ASTER svt;

    void (*start_state)(void *self_, RngState *s, Elem *S);
    bool (*transition) (void *self_, RngState *s,
                        Elem *S, Elem *A, real *R, Elem *nS,
                        uint t);
    bool (*is_terminal)(void *self_, Elem *S, uint t);
} EnvironmentVT;

typedef struct {
    EnvironmentVT VT_PTR_ASTER vt;
    uint flags;
} Environment;



// -------------------- MountainCar : Environment -------------

// According to Replacing trace Sutton Singh 1996

typedef struct {
    Environment super;
} MountainCar; // literally just a vtable pointer lol

#define MC_FL_STOPSHORT 1
#define MC_STOPSHORT_MAX_T 2000

extern inline void MountainCar_start_state(void *self_, RngState *s, Elem *S);
extern inline bool MountainCar_is_terminal(void *self_, Elem *S, uint t);
bool MountainCar_transition(void *self_, RngState *s,
                            Elem *S, Elem *A, real *R, Elem *nS,
                            uint t);

#define FROM_UNIT(x,a,b) ((a) + (x)*(b-a))
#define TO_UNIT(x,a,b) (((x)-(a))/((b)-(a)))

#define MC_LEFT  -1.2f
#define MC_RIGHT 0.5f
#define MC_MAXSPEED 0.07f

#define MC_POS_FROM_UNIT(x) FROM_UNIT((x),MC_LEFT,MC_RIGHT)
#define MC_POS_TO_UNIT(x) TO_UNIT((x),MC_LEFT,MC_RIGHT)
#define MC_SPEED_FROM_UNIT(x) FROM_UNIT((x),-MC_MAXSPEED,MC_MAXSPEED)
#define MC_SPEED_TO_UNIT(x) TO_UNIT((x),-MC_MAXSPEED,MC_MAXSPEED)

EnvironmentVT MountainCar_vt = {
    // need struct compound literal if VT_COPY, need init list otherwise.
    .svt = VT_COPY_SW(&(EnvironmentSVT),) {
        .name = "Mountain Car",
        .state_space = {
            .nfactors = 2,
            .factors = (SimpleSet[]){
                {.type='r', .as.range={.a=MC_LEFT,  .b=MC_RIGHT}},
                {.type='r', .as.range={.a=-MC_MAXSPEED, .b=MC_MAXSPEED}},
            }
        },
        .fixed_action_space = {
            .nfactors = 1,
            .factors = (SimpleSet[]){
                {.type='d', .as.discrete={.n=3}}
            }
        },
        .action_space_is_fixed = true,
        .action_space_is_discrete = true,
        .deterministic_transition = true,
    },
    .start_state = MountainCar_start_state,
    .transition = MountainCar_transition,
    .is_terminal = MountainCar_is_terminal,
};

// can be bad because A->MC_POS would be valid when it shouldnt
// makes it much easier to read however
#define MC_POS   x[0].r
#define MC_SPEED x[1].r
#define MC_ACCEL x[0].i

extern inline
void MountainCar_start_state(void *self_, RngState *s, Elem *S) {
    S->MC_POS = rand_uniform(s, -0.6f, -0.4f);
    S->MC_SPEED = 0.f;
}

extern inline
bool MountainCar_is_terminal(void *self_, Elem *S, uint t) {
    MountainCar *self = self_;
    return S->MC_POS >= MC_RIGHT
        || ((self->super.flags & MC_FL_STOPSHORT)
            && t > MC_STOPSHORT_MAX_T);
}

bool MountainCar_transition(void *self_, RngState *s,
                            Elem *S, Elem *A, real *R, Elem *nS,
                            uint t) {
    nS->MC_SPEED = S->MC_SPEED;
    nS->MC_SPEED += (((int)A->MC_ACCEL) - 1) * 0.001f; // drive force
    nS->MC_SPEED -= cosf(3.0f * S->MC_POS) * 0.0025f; // weight
    nS->MC_SPEED = rclamp(nS->MC_SPEED, -MC_MAXSPEED, MC_MAXSPEED);
    nS->MC_POS = S->MC_POS + nS->MC_SPEED; // position update
    if (nS->MC_POS < MC_LEFT) { // hit left wall
        nS->MC_POS = MC_LEFT;
        if (nS->MC_SPEED < 0.f)
            nS->MC_SPEED = 0.f;
    } else if (nS->MC_POS >= MC_RIGHT) {
        nS->MC_POS = MC_RIGHT;
        *R = 0.f;
    }
    *R = -1.f;
    return MountainCar_is_terminal(self_, nS, t);
}

extern inline
void make_MountainCar(MountainCar *self, uint flags) {
    self->super.vt = VT_PTR_AMPER MountainCar_vt;
    self->super.flags = flags;
}

extern inline
void free_MountainCar(MountainCar *self) {}

// convenience wrapper
extern inline
float MC_Qtheta(QFn *q, float pos, float speed, uint action) {
    SimpleElem A_accel = {.i = action},
        S_x[2] = {{.r = pos}, {.r = speed}};
    Elem A = {.x = &A_accel}, S = {.x = &S_x[0]};
    return Qtheta(q, &S, &A);
}
