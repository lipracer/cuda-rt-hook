CC:=gcc
CXX:=g++

all: main

xdnn: xdnn.cc 
	${CXX} xdnn.cc -g -shared -fpic -o libxdnn.so


xblas: xblas.cc xdnn
	${CXX} xblas.cc -g -shared -fpic -o libxblas.so -Wl,-rpath=./ -L. -lxdnn


xpu_kernel: xpu_kernel.cc 
	${CXX} xpu_kernel.cc -g -shared -fpic -o libxpu_kernel.so

xdnn_pytorch: xdnn_pytorch.cc xblas xpu_kernel
	${CXX} xdnn_pytorch.cc -g -shared -fpic -o libxdnn_pytorch.so -Wl,-rpath=./ -L. -lxpu_kernel -lxblas

main: main.cc xdnn_pytorch xblas
	${CXX} main.cc -g -o main -Wl,-rpath=./ -L.  -lxblas -lxdnn_pytorch

clean:
	rm -f *.so
	rm -f *.a
	rm -f main

test_main:
	./main

test_pytorch:
	python pytorch.py