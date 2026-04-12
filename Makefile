
LIB=-lm -lopenblas -lallegro -lallegro_primitives -lallegro_font
CFLAGS=-std=c17 -Wall -Wpedantic -fopenmp
EXTRA_FLAGS=
DEBUG_FLAGS=

MONOLITHIC_MAIN = main.c
main: *.c
	gcc $(MONOLITHIC_MAIN) -O2 -ffast-math $(CFLAGS) $(EXTRA_FLAGS) $(LIB) -o $@

mainf: *.c
	gcc $(MONOLITHIC_MAIN) -DNDEBUG -O3 -ffast-math -mtune=native -march=native \
		$(CFLAGS) $(EXTRA_FLAGS) $(LIB) -o $@

maind: *.c
	gcc $(MONOLITHIC_MAIN) -O0 -g $(DEBUG_FLAGS) $(CFLAGS) $(EXTRA_FLAGS) $(LIB) -o $@

mainprof: *.c
	gcc $(MONOLITHIC_MAIN) -O2 $(CFLAGS) $(EXTRA_FLAGS) $(LIB) -pg -o $@

clean:
	rm -f main mainf maind mainprof
