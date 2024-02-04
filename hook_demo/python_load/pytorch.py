import ctypes

xblas = ctypes.CDLL(f'./libxblas.so', mode=ctypes.RTLD_GLOBAL)
torch = ctypes.CDLL(f'./libxdnn_pytorch.so')

xblas.xdnn_xblas_add()
torch.xdnn_pytorch_add()