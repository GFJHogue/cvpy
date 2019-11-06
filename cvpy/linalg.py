import cv2 as cv

import cvpy as cp


def inv(a):
    if not isinstance(a, cp.Mat):
        raise TypeError("Must use Mat objects for inv()")
    elif not cp.CV_[a.dtype.name] in [cv.CV_32F, cv.CV_64F]:
        raise TypeError("Must use float32 or float64 dtype for inv()")
    shape = (a.shape[1], a.shape[0])
    out = cp.Mat(shape, dtype=a.dtype, UMat=a.UMat)
    if shape[0] == shape[1]:
        decompType = cv.DECOMP_LU
    else:
        decompType = cv.DECOMP_SVD
    cv.invert(a._, dst=out._, flags=decompType)
    return out


def lstsq(a, b, out=None):
    if out is None:
        shape = (a.shape[1], b.shape[1])
        out = cp.Mat(shape, dtype=a.dtype, UMat=a.UMat)
    if a.shape[0] == a.shape[1]:
        decompType = cv.DECOMP_LU
    else:
        decompType = cv.DECOMP_SVD
    isNonSingular, _ = cv.solve(a._, b._, dst=out._, flags=decompType)
    return out
