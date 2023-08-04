CC = gcc
USE_MP = 0
CFLAGS = -O3 -Wall -I/usr/include
LDFLAGS = -L/usr/lib -lm
ifeq ($(USE_MP), 1)
CFLAGS = -O3 -Wall -fopenmp -I/usr/include -DUSE_MP
LDFLAGS = -L/usr/lib -lomp -lm -DUSE_MP
ifeq ($(shell uname), Darwin)
CFLAGS = -O3 -Wall -I/opt/homebrew/opt/libomp/include -DUSE_MP
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp -lm -DUSE_MP
endif
endif

all: dynamic

dynamic: matrix.o nn.o
	$(CC) -shared -o libsimplematrix.so matrix.o nn.o $(LDFLAGS)

test: test.o matrix.o
	$(CC) -o test test.o matrix.o $(LDFLAGS)

test.o: test.c matrix.h
	$(CC) -c test.c $(CFLAGS)

static: matrix.o nn.o
	ar rcs libsimplematrix.a matrix.o nn.o

nn.o: nn.c nn.h matrix.h
	$(CC) -c nn.c $(CFLAGS)

matrix.o: matrix.c matrix.h
	$(CC) -c matrix.c $(CFLAGS)

clean:
	rm -rf *.o *.a test