import cv2 as cv
import numpy as np

from timeit import default_timer as timer

size = (10000, 10000)
type = np.float32

A = np.ones(size, type)
B = 2*np.ones(size, type)
C = np.zeros(size, type)

start = timer()
np.matmul(A, B, C)
end = timer()
print(end - start)
print(C)

A = cv.UMat(np.ones(size, type))
B = cv.UMat(2*np.ones(size, type))
C = cv.UMat(np.zeros(size, type))

start = timer()
cv.gemm(A, B, 1., None, 0., C)
end = timer()
print(end - start)
print(C)
