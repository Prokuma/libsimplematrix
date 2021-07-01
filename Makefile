test: test.o matrix.o
	cc -o test test.o matrix.o -lm

test.o: test.c matrix.h
	cc -c test.c

static: matrix.o nn.o
	ar rcs libsimplematrix.a matrix.o nn.o

nn.o: nn.c nn.h matrix.h
	cc -c nn.c

matrix.o: matrix.c matrix.h
	cc -c matrix.c

clean:
	rm -rf *.o *.a test
