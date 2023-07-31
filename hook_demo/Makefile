CC:=gcc
CXX:=g++

all: main_share main_static

main_share: libfoo.so libbar.so
	${CXX} main.cc -g -o main_share -Wl,-unresolved-symbols=ignore-in-shared-libs,-rpath=./ -L. -lfoo -lbar

libfoo.so: foo.cc
	${CXX} -g -shared -fpic foo.cc -o libfoo.so

libbar.so: bar.cc
	${CXX} -g -shared -fpic bar.cc -o libbar.so


main_static: libfoo.a libbar.a
	${CXX} -DSTATIC_LIBRARY main.cc -g -o main_static libfoo.a libbar.a

libfoo.a: foo.cc
	${CXX} -DSTATIC_LIBRARY -g -c -c foo.cc -o libfoo.a

libbar.a: bar.cc
	${CXX} -DSTATIC_LIBRARY -fvisibility=hidden -g -c bar.cc -o libbar.a

clean:
	rm -f *.so
	rm -f *.a
	rm -f main_share
	rm -f main_static

test_share:
	./main_share

test_static:
	./main_static