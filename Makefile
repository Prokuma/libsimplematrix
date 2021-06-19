test: test.o matrix.o
	cc -o test test.o matrix.o -lm

test.o: test.c matrix.h
	cc -c test.c

matrix.o: matrix.c matrix.h
	cc -c matrix.c

clean:
	rm -rf *.o test
