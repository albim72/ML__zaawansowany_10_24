from numba import jit,cuda
import numpy as np
from timeit import default_timer as timer


def cpufunction(a):
    for i in range(1_000):
        a[i] += 1

@jit(nopython=True)
def gpufunction(a):
    for i in range(1_000):
        a[i] += 1


if __name__ == '__main__':
    n = 1_000
    a = np.ones(n,dtype=np.float64)
    print(type(a))

    start = timer()
    cpufunction(a)
    print(f"czas działania na CPU: {timer() - start} s")

    start = timer()
    gpufunction(a)
    print(f"czas działania na GPU: {timer() - start} s")
