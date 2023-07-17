
main: libfoo.so libbar.so
	g++ main.cc -g -o main -Wl,-unresolved-symbols=ignore-in-shared-libs,-rpath=./ -L. -lfoo -lbar

libfoo.so: foo.cc
	g++ -g -shared -fpic foo.cc -o libfoo.so

libbar.so: bar.cc
	g++ -g -shared -fpic bar.cc -o libbar.so

clean:
	rm -f *.so
	rm -f main

test:
	./main