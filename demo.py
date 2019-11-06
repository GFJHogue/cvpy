import timeit

import numpy as np

import cvpy as cp


A = np.array([
    [ 0.,  1.],
    [ 1.,  1.],
    [ 2.,  1.],
    [ 3.,  1.]
], dtype=cp.uint8)
An = cp.Mat(A.shape, mat=A, UMat=False)
Au = cp.Mat(A.shape, mat=A, UMat=True)
B = np.array([[-1, 0.2, 0.9, 2.1]], dtype=cp.float32).T
Bn = cp.Mat(B.shape, mat=B, UMat=False)
Bu = cp.Mat(B.shape, mat=B, UMat=True)
Cn = cp.Mat((2, 1), dtype=cp.float32, UMat=False)
Cu = cp.Mat((2, 1), dtype=cp.float32, UMat=True)

np_results = [None]
cp_n_results = [None]
cp_u_results = [None]

def np_linalg_lstsq(res):
    res[0] = np.linalg.lstsq(A, B, rcond=None)[0]
def cp_n_linalg_lstsq(res):
    res[0] = cp.linalg.lstsq(An, Bn)
def cp_u_linalg_lstsq(res):
    res[0] = cp.linalg.lstsq(Au, Bu)

def np_addi(A):
    A += 1
def cp_n_addi(An):
    An += 1
def cp_u_addi(Au):
    Au += 1


'''print(timeit.timeit("np_linalg_lstsq(np_results)", globals=globals()))
print(np_results[0])
print(timeit.timeit("cp_n_linalg_lstsq(cp_n_results)", globals=globals()))
print(cp_n_results[0])
print(timeit.timeit("cp_u_linalg_lstsq(cp_u_results)", globals=globals()))
print(cp_u_results[0])'''

print(timeit.timeit("np_addi(A)", globals=globals()))
print(timeit.timeit("cp_n_addi(An)", globals=globals()))
print(timeit.timeit("cp_u_addi(Au)", globals=globals()))
