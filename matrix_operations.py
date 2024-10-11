import numpy as np
import psutil
import time
from memory_profiler import memory_usage


def multiplyBlock(A,B,C,N,blockSize,rowBlock,colBlock,kBlock):
    for i in range(rowBlock,min((rowBlock+blockSize),N),1):
        for j in range(colBlock,min((colBlock+blockSize),N),1):
            suma=0
            for k in range(kBlock,min((kBlock+blockSize),N),1):
                suma+=A[i,k]*B[k,j]
            C[i,j]+=suma
    return C

def matrix_multiply(A,B,C,N,blockSize):
    if  A.shape[1] == B.shape[0]:
        for i in range(0,N,blockSize):
            for j in range(0,N,blockSize):
                for k in range(0,N,blockSize):
                  C= multiplyBlock(A,B,C,N,blockSize,i,j,k)
        return C
    else:
        return "Sorry, cannot multiply A and B."

def track_memory(func, *args):
    mem_usage = memory_usage((func, args))
    print(f"Memory usage: {max(mem_usage)} MB")
    return mem_usage


def track_cpu(func, *args):
    cpu_percent_before = psutil.cpu_percent(interval=1)
    start_time = time.time()

    func(*args)

    end_time = time.time()
    cpu_percent_after = psutil.cpu_percent(interval=1)

    print(f"Initial CPU usage: {cpu_percent_before}%")
    print(f"Final CPU usage: {cpu_percent_after}%")
    print(f"Execution time: {end_time - start_time} seconds")
