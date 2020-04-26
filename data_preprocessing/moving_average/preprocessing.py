import numpy as np

#preprocessing functions
def minmax_scaler(x):
    if max(x) == min(x):
        return x, min(x), max(x)
    else:
        scaled_x = (x - min(x))/(max(x) - min(x))
        return scaled_x, min(x), max(x)

def piecewise_log(arr):
    arr[arr == 0] = 1
    return np.log(arr)
