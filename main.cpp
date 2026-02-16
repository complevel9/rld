#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <allegro5/allegro.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_primitives.h>
#include <allegro5/allegro_opengl.h>

#include "nd.hpp"
#include "rl.hpp"
#include "mountaincar.hpp"
#include "valuefn.hpp"
#include "sarsa.hpp"

#define szp(x) printf("sizeof " #x " %lu\n", sizeof(x))

using namespace std;

uint range_splitter_clamp(float x, float a, float b, uint num_splits)
{
    return (uint)((clamp(x, a, b) - a) / (b - a) * num_splits);
}

uint range_splitter_zero(float x, float a, float b, uint num_splits)
{
    if ((x < a) | (x > b))
        return num_splits; // invalid x so return invalid index
    return (uint)((x - a) / (b - a) * num_splits);
}


struct MCAggregateQFn : QFn<MC_Env> {
    uint x_splits, vx_splits;
    NdArray<float> theta_as_nd;
    MCAggregateQFn(uint x_splits_, uint vx_splits_)
        : x_splits(x_splits_), vx_splits(vx_splits_),
        theta_as_nd(&theta, 3, x_splits_, vx_splits_, 3)
    { }

    uint SA_to_flat(MC_State *s, MC_Action *a)
    {
        return theta_as_nd.flatpos(
            range_splitter_clamp(s->x, MC_LEFT, MC_RIGHT, x_splits),
            range_splitter_clamp(s->vx, -MC_SPEED_MAX, MC_SPEED_MAX,
                                 vx_splits),
        *a);
    }

    real Q_theta(MC_State *s, MC_Action *a)
    {
        return theta[SA_to_flat(s, a)];
    }

    void dQ(MC_State *s, MC_Action *a, Vec<float> *g)
    {
        g->set_all(0.0f);
        (*g)[SA_to_flat(s, a)] = 1.0f;
    }

    void dQ_addto(MC_State *s, MC_Action *a, Vec<float> *z)
    {
        (*z)[SA_to_flat(s, a)] += 1.0f;
    }
};
//struct tt {int a;};
struct MCDirectFeatureLinearQFn : QFn<MC_Env> {
    MCDirectFeatureLinearQFn() : QFn(4)
    {}

    real Q_theta(MC_State *s, MC_Action *a)
    {
        return s->x*theta[0] + s->vx*theta[1] + ((int)(*a) - 1)*theta[2];
    }

    void dQ(MC_State *s, MC_Action *a, Vec<float> *g)
    {
        (*g)[0] = s->x;
        (*g)[1] = s->vx;
        (*g)[2] = ((int)*a) - 1;
    }

    void dQ_addto(MC_State *s, MC_Action *a, Vec<float> *z)
    {
        (*z)[0] += s->x;
        (*z)[1] += s->vx;
        (*z)[2] += ((int)*a) - 1;
    }
};

template<class EnvI, class QFnI> struct Stupid_Algo : Algo<EnvI,QFnI> {
    typedef typename EnvI::State State;
    typedef typename EnvI::Action Action;

    uint t;
    void begin_episode(QFnI *Q) { t = 0; }
    Action choose_action(State *s) {
        t++; if (t <= 29) return MC_forward; else if (t <= 60 )return MC_backward; return MC_forward;}
    void update(QFnI *Q, uint t, State *s, Action *a, real r, State *ss,
                bool is_terminal) {};
};

MC_Env mcenv;
MCAggregateQFn Q_agg(3, 5);
//MCDirectFeatureLinearQFn Q;
EpsilonGreedy<MC_Env, MCAggregateQFn> epgreedy;
SarsaLambda_Algo<MC_Env, MCAggregateQFn, EpsilonGreedy<MC_Env,MCAggregateQFn>>        sarsa_lambda(0.02, 0.99, 0.95, &Q_agg, &epgreedy);
NatSarsaLambda_Algo<MC_Env, MCAggregateQFn, EpsilonGreedy<MC_Env,MCAggregateQFn>> nat_sarsa_lambda(0.02, 0.99, 0.95, &Q_agg, &epgreedy);
uint ep = 0, t = 0;

auto env = &mcenv;
//auto algo = &sarsa_lambda;
auto algo = &nat_sarsa_lambda;
auto Q = algo->Q;

typedef MC_State State;
typedef MC_Action Action;

State s;
bool is_terminal = false;

ALLEGRO_TIMER *timer = NULL;
ALLEGRO_EVENT_QUEUE *queue = NULL;
ALLEGRO_DISPLAY *display = NULL;
ALLEGRO_FONT *font = NULL;

bool logmc = false;


void draw_grid(int xs, int ys, uint rx, uint ry, int cellsz)
{
	for(uint y = 0; y <= ry; y++)
		al_draw_line(xs+0.5f, ys+y*cellsz+0.5f, xs+rx*cellsz+0.5f,
			ys+y*cellsz+0.5f, al_map_rgb(255, 0, 0), 0.f);
	for(uint x = 0; x <= rx; x++)
		al_draw_line(xs+x*cellsz+0.5f, ys+0.5f, xs+x*cellsz+0.5f,
			ys+ry*cellsz+0.5f, al_map_rgb(255, 0, 0), 0.f);
}
float edge(float a, float b, float frac)
{
	return (frac - a) / (b - a);
}
#define CLAMP(a,b,v) ((v) < (a) ? (a) : (v) > (b) ? (b) : v)

ALLEGRO_COLOR heat_palette(float min, float max, float val)
{
	float x = (val-min) / (max-min);
	x = CLAMP(0.f, 1.f, x);
	float r = edge(.35f, .61f, x);
	float gx = fabsf(x-.8f);
	float bx = fabsf(x-.625f);
	float g = -16.f * gx * (gx - .5f);
	float b = -16.f * (bx-.25f) * (bx-.5f) + .75f;
	return al_map_rgb_f(r, g, b);
}

float mc_frac(float x) {
    return (x - MC_LEFT) / (MC_RIGHT - MC_LEFT);
}

void draw_Q(int xss, int yss, int cs)
{
    uint xim  = Q_agg.theta_as_nd.dim[0];
    uint vxim = Q_agg.theta_as_nd.dim[1];
    for (uint xi = 0; xi < xim; xi++)
    {
        for (uint vxi = 0; vxi < vxim; vxi++)
        {
            int xs = xss + (xi) * cs;
            int ys = yss + (vxi) * (3 * cs + 10);
            for (uint a = 0; a < 3; a++)
            {
                int x = 0, y = a;
                float q = Q_agg.theta_as_nd.at(xi, vxi, a);
                ALLEGRO_COLOR rect_color = heat_palette(-300, 0, q);
				ALLEGRO_COLOR text_color = rect_color.r + rect_color.g + rect_color.b > 0.5
					? al_map_rgb(0,0,0)
					: al_map_rgb(255,255,255);
				al_draw_filled_rectangle(xs+x*cs+.5f, ys+y*cs+.5f,
					xs+(x+1)*cs+.5f, ys+(y+1)*cs+.5f, rect_color);
				al_draw_textf(font, text_color,
					xs+x*cs+1.f, ys+y*cs+2.f, ALLEGRO_ALIGN_LEFT,
					"%.2f", q);
            }
            draw_grid(xs, ys, 1, 3, cs);
        }
    }
}

void new_ep()
{
    s = env->start_state();
    env->begin_episode();
    algo->begin_episode(&s);
    t = 0;
}

void one_ep()
{
    logmc = false;
    run_episode<MC_Env, MCAggregateQFn>(&mcenv, algo, false);
    ep++;
}

void setup()
{
    srand(0);
    epgreedy.Q = &Q_agg;
    epgreedy.epsilon = 0.02;
    epgreedy.env = &mcenv;
    epgreedy.random_tiebreak = true;
    srand(0);
    Q_agg.theta.set_all(1.0f); // optimistic initialisation
    for (uint i = 0; i < 0; i++)
        one_ep();
    new_ep();
}


void draw()
{
    //puts("hi");
    Action a = algo->choose_action(&s);
    real r;
    State ss;
    is_terminal = env->transition(t, &s, &a, &r, &ss);
    algo->update(t, &s, &a, r, &ss, is_terminal);
    if (is_terminal)
    {
        ep++;
        printf("SL:  ep %u took %u steps\n", ep, t);
        //one_ep();
        new_ep();
        return;
    }
    s = ss;
    t++;

    al_clear_to_color(al_map_rgb(0, 0, 0));
    draw_Q(10, 10, 40);
    al_draw_textf(font, al_map_rgb(255, 255, 255), 400, 12, 0, "episode: %u", ep);
	al_draw_textf(font, al_map_rgb(255, 255, 255), 400, 24, 0, "time steps: %u", t);
    al_draw_line(400, 100, 700, 100, al_map_rgb(255, 255, 0), 0);
    al_draw_circle(400 + (700-400)*mc_frac(s.x), 100, 5, al_map_rgb(0, 255, 0), 0);
    al_draw_line((400 + 700)/2, 120,
                 (400 + 700)/2 + (700-400)/4 * (((int)algo->action)-1),
                 120, al_map_rgb(255, 255, 0), 0);
    for (uint i = 0; i < 5; i++)
    {
        uint x = 400 + (700-400)*i/4;
        al_draw_line(x, 90, x, 110,
                     al_map_rgb(255, 255, 255), 0);
    }
    al_flip_display();
    //logmc = true;

}

int main(int argc, char **argv)
{
    szp(NdArray<float>);

    printf("Q.xs %u; Q.vxs %u\n", Q_agg.x_splits, Q_agg.vx_splits);
    Q_agg.theta_as_nd.print_dim();
    setup();
//    Q_agg.theta.set_all(1.0f); // optimistic initialisation
//    for (uint ep = 0; ep < 20; ep++)
//    {
//        uint steps = run_episode<MC_Env, MCAggregateQFn>(&mcenv, &sarsa_lambda, false);
//        printf("SL:  ep %u took %u steps\n", ep, steps);
//    }
//    for (uint ep = 0; ep < 20; ep++)
//    {
        //epgreedy.epsilon = 1.0f / (2+ep);
//        uint steps = run_episode<MC_Env, MCAggregateQFn>(&mcenv, &nat_sarsa_lambda, false);
//        printf("NSL: ep %u took %u steps\n", ep, steps);
//        for (uint j = 0; j < Q.vx_splits; j++)
//        {
//            for (uint i = 0; i < Q.x_splits; i++)
//            {
//                uint k = i*Q.vx_splits*3 + j*3;
//                printf("%7.1f|%7.1f|%7.1f", Q.theta[k],Q.theta[k+1],Q.theta[k+2]);
//                if (i < 3) printf("\t");
//            }
//            printf("\n");
//
//        }
//    }
    // allegro display

#define do_die(do,die_msg) if (!(do)) {fputs(die_msg, stderr);return 1;}
#define init_die(do,die_msg) do_die((do), "Couldn't init " die_msg "!")
	init_die(al_init(), "allegro")
	init_die(al_init_primitives_addon(), "primitives addon")
	init_die(al_install_keyboard(), "keyboard")

#define FPS 60
	timer = al_create_timer(1.0 / FPS);
	init_die(timer, "timer")
	queue = al_create_event_queue();
	init_die(queue, "event queue");

	display = al_create_display(800, 800);
	init_die(display, "display")
	al_set_window_title(display, "woah salsa");

	font = al_create_builtin_font();
	init_die(font, "font")

	al_register_event_source(queue, al_get_keyboard_event_source());
	al_register_event_source(queue, al_get_display_event_source(display));
	al_register_event_source(queue, al_get_timer_event_source(timer));

	bool redraw = false;
	bool done = false;
	ALLEGRO_EVENT event;
	al_start_timer(timer);
	draw();
	while(!done)
	{
		al_wait_for_event(queue, &event);
		switch(event.type)
		{
		case ALLEGRO_EVENT_TIMER:
			redraw = true;
			break;
		case ALLEGRO_EVENT_KEY_DOWN:
			if(event.keyboard.keycode == ALLEGRO_KEY_ESCAPE)
				done = true;
			break;
		case ALLEGRO_EVENT_DISPLAY_CLOSE:
			done = true;
			break;
		}
		if(redraw && al_is_event_queue_empty(queue))
		{
			draw();
			redraw = false;
		}
	}


    /*
    szp(State);
    szp(Action);
    szp(AgentVT);
    szp(EnvVT);
    szp(ExperimentSetup);
    szp(size_t);
    szp(long int);
    szp(long long int);

    */
}
