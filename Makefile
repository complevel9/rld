
LIB=-lm -lopenblas -lallegro -lallegro_primitives -lallegro_font
CFLAGS=-std=c17 -Wall -Wpedantic -fopenmp
EXTRA_FLAGS=
DEBUG_FLAGS=

main: main.c
	gcc $^ -O3 -ffast-math -s $(CFLAGS) $(EXTRA_FLAGS) $(LIB) -o $@

maind: main.c
	gcc $^ -O0 -g $(DEBUG_FLAGS) $(CFLAGS) $(EXTRA_FLAGS) $(LIB) -o $@

mainprof: main.c
	gcc $^ -O2 $(CFLAGS) $(EXTRA_FLAGS) $(LIB) -pg -o $@

clean:
	rm -f main maind mainprof
