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


// -------------------- Flip ----------------------------------

typedef struct {
    Environment super;
    uint steps_till_flip;
    uint win_action;
} Flip;

#define STEPS_PER_FLIP 2000

void Flip_start_state(void *self_, RngState *s, Elem *S) {
    S->x[0].i = 0;
}

bool Flip_is_terminal(void *self_, Elem *S, uint t) {
    return t > 0 || S->x[0].i == 1;
}

bool Flip_transition(void *self_, RngState *s,
                     Elem *S, Elem *A, real *R, Elem *nS,
                     uint t) {
    Flip *self = self_;
    nS->x[0].i = 1;
    *R = self->win_action == A->x[0].i ? 1.f : 0.f;
    // printf("chose action %u with reward %.f\n", t, A->x[0].i, *R);
    // getchar();
    if (self->steps_till_flip) {
        self->steps_till_flip--;
    } else {
        // printf("flip!!!!!!!!\n");
        self->steps_till_flip = STEPS_PER_FLIP;
        self->win_action = 1 - self->win_action;
    }
    return true;
}

EnvironmentVT Flip_vt = {
    .svt = VT_COPY_SW(&(EnvironmentSVT),) {
        .name = "Flip",
        .state_space = {
            .nfactors = 1,
            .factors = (SimpleSet[]){{.type='d', .as.discrete.n=1}}
        },
        .fixed_action_space = {
            .nfactors = 1,
            .factors = (SimpleSet[]){{.type='d', .as.discrete.n=2}}
        },
        .action_space_is_fixed = true,
        .action_space_is_discrete = true,
        .deterministic_transition = true,
    },
    .start_state = Flip_start_state,
    .transition = Flip_transition,
    .is_terminal = Flip_is_terminal,
};

void make_Flip(Flip *self) {
    self->super.vt = VT_PTR_AMPER Flip_vt;
    self->steps_till_flip = STEPS_PER_FLIP;
    self->win_action = 0;
}

void free_Flip(Flip *self) {}

// -------------------- MountainCar ---------------------------

// According to Replacing trace Sutton Singh 1996, but with max timesteps

typedef struct {
    Environment super;
} MountainCar;

#define MC_FL_STOPSHORT 1
#define MC_STOPSHORT_MAX_T 10000
#define MC_MAX_T 10000

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
                {.type='d', .as.discrete.n=3}
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
    if (t > MC_MAX_T)
        return true;
    if (t > MC_STOPSHORT_MAX_T && (self->super.flags & MC_FL_STOPSHORT))
        return true;
    return S->MC_POS >= MC_RIGHT;
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
