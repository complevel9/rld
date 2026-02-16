#ifndef HEADER_SARSA_HPP_
#define HEADER_SARSA_HPP_

#include "rl.hpp"

#include "common.hpp"

#include <cstdlib>

bool bernouli(float pr) {
    return (rand()/(float)RAND_MAX) < pr;
}

#define FLOAT_TOLERANCE 0.0000002f
bool greedy = false;

template<class EnvI, class QFnI>
struct EpsilonGreedy {
    typedef typename EnvI::State State;
    typedef typename EnvI::Action Action;

    QFnI *Q;
    EnvI *env;
    float epsilon;
    bool random_tiebreak;
    Action choose_action(State *s)
    {
        uint n = env->num_actions_at(s);
        uint a = 0;
        if (epsilon <= 0.0f || bernouli(epsilon))
        {
            // explore
            return rand() % n; // ignore RAND_MAX bias, it's small
        }

        uint tie_pick = 0;
        float max_value = -999999999999.0f;

        // greedy
        greedy = true;
        for (uint i = 0; i < n; i++)
        {
            float value = Q->Q_theta(s, &i);
            if (value > max_value + FLOAT_TOLERANCE)
            {
                a = i;
                max_value = value;
                tie_pick = 1;
            }
            else if (value >= max_value - FLOAT_TOLERANCE)
            {
                tie_pick++;
            }
        }

        if (tie_pick == 0)
        {

            sigtrap();
            for (uint i = 0; i < n; i++)
            {
                float value = Q->Q_theta(s, &i);
                if (value > max_value + FLOAT_TOLERANCE)
                {
                    a = i;
                    max_value = value;
                    tie_pick = 1;
                }
                else if (value >= max_value - FLOAT_TOLERANCE)
                {
                    tie_pick++;
                }
            }
        }

        if (random_tiebreak && tie_pick == 1)
            return a;

        // random tie breaking
        tie_pick = rand() % tie_pick;
        for (uint i = 0; i < n; i++)
        {
            float value = Q->Q_theta(s, &i);
            if ((max_value - FLOAT_TOLERANCE <= value)
              & (value <= max_value + FLOAT_TOLERANCE))
            {
                if (tie_pick == 0)
                    return i;
                tie_pick--;
            }
        }
        fputs("tiebreak fail", stderr);
        exit(5);
        return a;
    }
};

template<class EnvI, class QFnI, class Policy> struct SarsaLambda_Algo : Algo<EnvI, QFnI> {
    typedef typename EnvI::State State;
    typedef typename EnvI::Action Action;

    float alpha;
    float gamma;
    float lambda;
    Action action;
    QFnI *Q;
    Policy *pi;
    Vec<float> z;
    SarsaLambda_Algo(float alpha_, float gamma_, float lambda_, QFnI *Q_,
                     Policy *pi_)
        : alpha(alpha_), gamma(gamma_), lambda(lambda_), Q(Q_),
          pi(pi_), z(Q_->theta.size)
    { }

    void begin_episode(State *s)
    {
        z.zero();
        action = pi->choose_action(s);
    }

    Action choose_action(State *s)
    {
        return action;
    }

    void update(uint t, State *s, Action *a, real r, State *ss,
                bool is_terminal)
    {
        float delta = r - Q->Q_theta(s, a);
        Q->dQ_addto(s, a, &z);
        if (is_terminal) {
            Q->theta.add_scaled(alpha * delta, &z);
            return;
        }
        action = pi->choose_action(s);
        delta += gamma * Q->Q_theta(ss, &action);
        Q->theta.add_scaled(alpha * delta, &z);
        z.scale(gamma * lambda);
    }
};

struct Mat : NdArray<float> {
    Mat(uint m_, uint n_) : NdArray(nullptr, 2, m_, n_) {}
    Mat(uint m_) : Mat(m_, m_) {}
    void set_diag_all(float v)
    {
        uint m = dim[1] < dim[0] ? dim[0] : dim[1];
        for (uint i = 0; i < m; i++)
            data[i + i*m] = v;
    }
    void print() {
        for (uint i = 0; i < dim[0]; i++)
        {
            for (uint j = 0;  j < dim[1]; j++)
            {
                float f = data[i*dim[1] + j];
                if (fabs(f) < 0.001f)
                    printf("(    ) ");
                else
                    printf("(%4.1f) ", f);
            }
            putchar('\n');
        }
    }
};

// A_{m x n} x v_{n x 1}
void Av(Mat *A, Vec<float> *v, Vec<float> *dest)
{
    uint m = A->dim[0];
    uint n = A->dim[1];
    if ((v->size != n) | (dest->size != m))
        exit(56);
    for (uint i = 0; i < m; i++)
    {
        dest->data[i] = 0.0f;
        for (uint j = 0; j < n; j++)
            dest->data[i] += A->data[i*n + j] * v->data[j];
    }
}

// A^T_{n x m}
void ATv(Mat *A, Vec<float> *v, Vec<float> *dest)
{
    uint m = A->dim[0];
    uint n = A->dim[1];
    if ((v->size != m) | (dest->size != n))
        exit(57);
    for (uint j = 0; j < n; j++)
    {
        dest->data[j] = 0.0f;
        for (uint i = 0; i < m; i++)
            dest->data[j] += A->data[j*m + i] * v->data[i];
    }
}

// uvT
void outer(Vec<float> *u, Vec<float> *v, Mat *dest)
{
    uint m = u->size;
    uint n = v->size;
    if ((dest->dim[0] != m) | (dest->dim[1] != n))
        exit(58);
    for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
            dest->data[i*n + j] = u->data[i] * v->data[j];
}

// A += alpha * uvT
void outer_scaled_addto(Vec<float> *u, Vec<float> *v, float scale, Mat *dest)
{
    uint m = u->size;
    uint n = v->size;
    if ((dest->dim[0] != m) | (dest->dim[1] != n))
        exit(58);
    for (uint i = 0; i < m; i++)
        for (uint j = 0; j < n; j++)
            dest->data[i*n + j] += scale * u->data[i] * v->data[j];
}

// uTv
float inner(Vec<float> *u, Vec<float> *v) {
    uint n = u->size;
    if (n != v->size)
        exit(59);
    float ret = 0.0f;
    for (uint i = 0; i < n; i++)
        ret += u->data[i] * v->data[i];
    return ret;
}

template<class EnvI, class QFnI, class Policy> struct NatSarsaLambda_Algo : Algo<EnvI, QFnI> {
    typedef typename EnvI::State State;
    typedef typename EnvI::Action Action;

    float alpha;
    float gamma;
    float lambda;
    Vec<float> elig, g;
    Vec<float> v1, v2; // scratch
    Mat G_inv;
    Action action;
    QFnI *Q;
    Policy *pi;
    NatSarsaLambda_Algo(float alpha_, float gamma_, float lambda_, QFnI *Q_,
                     Policy *pi_)
        : alpha(alpha_), gamma(gamma_), lambda(lambda_),
          elig(Q_->theta.size), g(Q_->theta.size),
          v1(Q_->theta.size), v2(Q_->theta.size),
          G_inv(Q_->theta.size),
          Q(Q_), pi(pi_)
    { }

    void begin_episode(State *s)
    {
        elig.zero();
        G_inv.set_diag_all(1.0f);
//        for (uint i = 0; i < G_inv.dim[0]*G_inv.dim[1]; i++)
//            printf("%.0f ", G_inv.data[i]);
        action = pi->choose_action(s);
    }

    Action choose_action(State *s)
    {
        return action;
    }

    void update(uint t, State *s, Action *a, real r, State *ss,
                bool is_terminal)
    {
        float delta = r - Q->Q_theta(s, a);
        if (!is_terminal)
        {
            action = pi->choose_action(s);
            delta += gamma * Q->Q_theta(ss, &action);
        }

        Q->dQ(s, a, &g);

        for (uint i = 0; i < elig.size; i++)
        {
            elig.data[i] *= gamma * lambda;
            elig.data[i] += g.data[i];
        }

        Av(&G_inv, &g, &v1); // v1 = G^{-1} g
        ATv(&G_inv, &g, &v2); // v2 = G^{-T} g
        float scale = -delta * delta / (1 + delta*delta*inner(&g, &v1));
        outer_scaled_addto(&v1, &v2, scale, &G_inv);
        Av(&G_inv, &elig, &v1); // v1 = G^{-1} e

#define floateq(a,b) ((a - FLOAT_TOLERANCE <= b) & (b <= a + FLOAT_TOLERANCE))

        G_inv.print();
        g.print();
        printf("delta=%f\n", delta);
//        printf("scale=%f\n", scale);
//        float d = inner(&v1, &elig);
//        if (d <= 0)
//        {
//            puts("WTF NOT POSITIVE DEFINITE");
//            for (uint j = 0; j < elig.size; j++)
//                printf("[%3u] %c %.5f , %.5f\n", j, floateq(elig[j], v1[j]) ? ' ' : '*', elig[j], v1[j]);
//            sigtrap();
//            puts("what");
//        }

//        for (uint i = 0; i < elig.size; i++)
//            if (!floateq(elig[i], v1[i]))
//            {
//                printf("wtf??\n");
//                for (uint j = 0; j < elig.size; j++)
//                    printf("[%3u] %c %.5f , %.5f\n", j, floateq(elig[j], v1[j]) ? ' ' : '*', elig[j], v1[j]);
//                getchar();
//                break;
//            }
        Q->theta.add_scaled(alpha * delta, &v1);
//        Q->theta.add_scaled(alpha * delta, &elig);
    }
};
/*      float delta = r - Q->Q_theta(s, a);
        Q->dQ_addto(s, a, &z);
        if (is_terminal) {
            Q->theta.add_scaled(alpha * delta, &z);
            return;
        }
        action = pi->choose_action(s);
        delta += gamma * Q->Q_theta(ss, &action);
        Q->theta.add_scaled(alpha * delta, &z);
        z.scale(gamma * lambda);
*/

#endif
