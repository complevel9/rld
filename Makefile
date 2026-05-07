CC=gcc
LIB=-lm -lopenblas -lallegro -lallegro_primitives -lallegro_font
CFLAGS=-std=c17 -Wall -Wpedantic -fopenmp
EXTRA_FLAGS=
DEBUG_FLAGS=
DD_FLAGS=-fsanitize=undefined,address

MONOLITHIC_MAIN = main.c
main: *.c
	$(CC) $(MONOLITHIC_MAIN) -O2 -ffast-math $(CFLAGS) $(EXTRA_FLAGS) $(LIB) -o $@

mainf: *.c
	$(CC) $(MONOLITHIC_MAIN) -O3 -ffast-math -DNDEBUG -mtune=native -march=native \
		$(CFLAGS) $(EXTRA_FLAGS) $(LIB) -o $@

maind: *.c
	$(CC) $(MONOLITHIC_MAIN) -O0 -g $(DEBUG_FLAGS) $(CFLAGS) $(EXTRA_FLAGS) $(LIB) -o $@

maindd: *.c
	$(CC) $(MONOLITHIC_MAIN) -O0 -g $(DEBUG_FLAGS) $(DD_FLAGS) $(CFLAGS) $(EXTRA_FLAGS) $(LIB) -o $@

mainprof: *.c
	$(CC) $(MONOLITHIC_MAIN) -O2 $(CFLAGS) $(EXTRA_FLAGS) $(LIB) -pg -o $@

clean:
	rm -f main mainf maind mainprof
