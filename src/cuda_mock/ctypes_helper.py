import ctypes

def convert_arg_list_of_str(list_of_str):
    N = len(list_of_str)
    arr = (ctypes.c_char_p * (N + 1))()
    
    arr[:-1] = [s.encode('utf-8') for s in list_of_str]
    arr[-1] = None
    return arr