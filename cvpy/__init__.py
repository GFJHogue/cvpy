import cv2 as cv
import numpy as np

import cvpy.linalg as linalg


uint8   = np.uint8
int8    = np.int8
uint16  = np.uint16
int16   = np.int16
int32   = np.int32
float32 = np.float32
float64 = np.float64

CV_ = {
    'uint8':   cv.CV_8U,
    'int8':    cv.CV_8S,
    'uint16':  cv.CV_16U,
    'int16':   cv.CV_16S,
    'int32':   cv.CV_32S,
    'float32': cv.CV_32F,
    'float64': cv.CV_64F
}

USAGE_DEFAULT                = cv.USAGE_DEFAULT
USAGE_ALLOCATE_HOST_MEMORY   = cv.USAGE_ALLOCATE_HOST_MEMORY
USAGE_ALLOCATE_DEVICE_MEMORY = cv.USAGE_ALLOCATE_DEVICE_MEMORY
USAGE_ALLOCATE_SHARED_MEMORY = cv.USAGE_ALLOCATE_SHARED_MEMORY


def _src2(src2, UMat=False):
    if isinstance(src2, Mat):
        if UMat:
            return src2._
        else:
            return src2.n
    return src2

def _mimicMat(mat):
    return Mat(mat.shape, dtype=mat.dtype, UMat=mat.UMat)

def _mimicMatZeros(mat):
    mat = Mat(mat.shape, dtype=mat.dtype, UMat=mat.UMat)
    cv.subtract(mat._, mat._, mat._)
    nan_to_num(mat, copy=False)
    return mat


class Mat:

    def __init__(
        self,
        shape, dtype=float64, buffer=None, offset=0, strides=None, order=None,
        mat=None, UMat=False, usageFlags=USAGE_DEFAULT
    ):
        self.UMat = UMat
        if mat is None:
            mat = np.ndarray(
                shape, dtype=dtype, buffer=buffer,
                offset=offset, strides=strides, order=order
            )
        self.shape = mat.shape
        self.dtype = mat.dtype
        if UMat:
            self._n = None
            self.u = cv.UMat(mat)
            self._ = self.u
        else:
            self._n = mat
            self.u = None
            self._ = self._n

    @property
    def n(self):
        if self.UMat:
            return self.u.get()
        else:
            return self._n
    def get(self):
        return self.n

    @property
    def T(self):
        out = _mimicMat(self)
        cv.transpose(self._, out._)
        return out

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def copy(self, mask=None):
        mask = _src2(mask, UMat=self.UMat)
        out = _mimicMat(self)
        cv.copyTo(self._, mask, out._)
        return out

    def fill(self, value):
        self -= self
        nan_to_num(self, copy=False)
        self += value
        return None

    def sort(self, axis=1, desc=False):
        if axis == 0:
            sortFlags = cv.SORT_EVERY_COLUMN
        elif axis == 1:
            sortFlags = cv.SORT_EVERY_ROW
        if desc:
            sortFlags += cv.SORT_DESCENDING
        else:
            sortFlags += cv.SORT_ASCENDING
        out = _mimicMat(self)
        cv.sort(self._, sortFlags, out._)
        del self._
        if self.UMat:
            del self.u
            self.u = out.u
            self._ = self.u
        else:
            del self._n
            self._n = out._n
            self._ = self._n
        return None

    def argsort(self, axis=1, desc=False):
        if axis == 0:
            sortFlags = cv.SORT_EVERY_COLUMN
        elif axis == 1:
            sortFlags = cv.SORT_EVERY_ROW
        if desc:
            sortFlags += cv.SORT_DESCENDING
        else:
            sortFlags += cv.SORT_ASCENDING
        out = Mat(self.shape, dtype=int32, UMat=self.UMat)
        cv.sortIdx(self._, sortFlags, out._)
        return out

    def max(self, mask=None):
        mask = _src2(mask, UMat=self.UMat)
        _, maxVal, _, _ = cv.minMaxLoc(self._, mask=mask)
        return maxVal

    def argmax(self, mask=None):
        mask = _src2(mask, UMat=self.UMat)
        _, _, _, maxLoc = cv.minMaxLoc(self._, mask=mask)
        return maxLoc

    def min(self, mask=None):
        mask = _src2(mask, UMat=self.UMat)
        minVal, _, _, _ = cv.minMaxLoc(self._, mask=mask)
        return minVal

    def argmin(self, mask=None):
        mask = _src2(mask, UMat=self.UMat)
        _, _, minLoc, _ = cv.minMaxLoc(self._, mask=mask)
        return minLoc

    def ptp(self, mask=None):
        mask = _src2(mask, UMat=self.UMat)
        minVal, maxVal, _, _ = cv.minMaxLoc(self._, mask=mask)
        return maxVal - minVal

    def clip(self, min=None, max=None, out=None):
        if out is None:
            out = self
        else:
            cv.add(self._, 0, dst=out._, dtype=CV_[out.dtype.name])
        if not min is None:
            mask = self < min
            cv.subtract(out._, out._, dst=out._, mask=mask._)
            cv.add(out._, min, dst=out._, mask=mask._)
        if not max is None:
            mask = self > max
            cv.subtract(out._, out._, dst=out._, mask=mask._)
            cv.add(out._, max, dst=out._, mask=mask._)
        return out

    def trace(self):
        tr = np.trim_zeros(cv.trace(self._), 'b')
        if len(tr) == 0:
            return (0.,)
        return tr

    def sum(self):
        sm = np.trim_zeros(cv.sumElems(self._), 'b')
        if len(sm) == 0:
            return (0.,)
        return sm

    def mean(self, mask=None):
        mask = _src2(mask, UMat=self.UMat)
        if self.ndim == 3 and self.shape[2] <= 4:
            shape = (1, self.shape[2])
        else:
            shape = (1, 1)
        mean = Mat(shape, dtype=float64, UMat=self.UMat)
        cv.meanStdDev(self._, mean=mean._, mask=mask)
        return mean

    def var(self, mask=None):
        vr = self.std(mask)
        vr *= vr
        return vr

    def std(self, mask=None):
        mask = _src2(mask, UMat=self.UMat)
        if self.ndim == 3 and self.shape[2] <= 4:
            shape = (1, self.shape[2])
        else:
            shape = (1, 1)
        stddev = Mat(shape, dtype=float64, UMat=self.UMat)
        cv.meanStdDev(self._, stddev=stddev._, mask=mask)
        return stddev

    def all(self):
        return cv.countNonZero(self._) == self.size

    def any(self):
        return cv.countNonZero(self._) > 0

    def __str__(self):
        if self.UMat:
            str = 'UMat:'
        else:
            str = 'Mat:'
        str += self.dtype.name
        str += '\n'
        return str+self.n.__str__()

    def __lt__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        out = Mat(self.shape, dtype=uint8, UMat=self.UMat)
        cv.compare(self._, src2, cv.CMP_LT, dst=out._)
        return out
    def __le__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        out = Mat(self.shape, dtype=uint8, UMat=self.UMat)
        cv.compare(self._, src2, cv.CMP_LE, dst=out._)
        return out
    def __gt__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        out = Mat(self.shape, dtype=uint8, UMat=self.UMat)
        cv.compare(self._, src2, cv.CMP_GT, dst=out._)
        return out
    def __ge__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        out = Mat(self.shape, dtype=uint8, UMat=self.UMat)
        cv.compare(self._, src2, cv.CMP_GE, dst=out._)
        return out
    def __eq__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        out = Mat(self.shape, dtype=uint8, UMat=self.UMat)
        cv.compare(self._, src2, cv.CMP_EQ, dst=out._)
        return out
    def __ne__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        out = Mat(self.shape, dtype=uint8, UMat=self.UMat)
        cv.compare(self._, src2, cv.CMP_NE, dst=out._)
        return out

    def __bool__(self):
        return self.n.__bool__()

    def __neg__(self):
        return self.__mul__(-1)
    def __pos__(self):
        return self
    def __abs__(self):
        out = _mimicMat(self)
        cv.absdiff(self._, 0, dst=out._)
        return out
    def __invert__(self):
        out = _mimicMat(self)
        cv.bitwise_not(self._, dst=out._)
        return out

    def __add__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        out = _mimicMat(self)
        cv.add(self._, src2, dst=out._, dtype=CV_[self.dtype.name])
        return out
    def __radd__(self, value):
        return self.__add__(value)
    def __iadd__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        cv.add(self._, src2, dst=self._, dtype=CV_[self.dtype.name])
        return self

    def __sub__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        out = _mimicMat(self)
        cv.subtract(self._, src2, dst=out._, dtype=CV_[self.dtype.name])
        return out
    def __rsub__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        out = _mimicMat(self)
        cv.subtract(src2, self._, dst=out._, dtype=CV_[self.dtype.name])
        return out
    def __isub__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        cv.subtract(self._, src2, dst=self._, dtype=CV_[self.dtype.name])
        return self

    def __mul__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        out = _mimicMat(self)
        cv.multiply(self._, src2, dst=out._, dtype=CV_[self.dtype.name])
        return out
    def __rmul__(self, value):
        return self.__mul__(value)
    def __imul__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        cv.multiply(self._, src2, dst=self._, dtype=CV_[self.dtype.name])
        return self

    def __truediv__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        if CV_[self.dtype.name] in [cv.CV_32F, cv.CV_64F]:
            out = _mimicMat(self)
        else:
            out = Mat(self.shape, dtype=float64, UMat=self.UMat)
        cv.divide(self._, src2, dst=out._, dtype=CV_[out.dtype.name])
        return out
    def __rtruediv__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        if CV_[self.dtype.name] in [cv.CV_32F, cv.CV_64F]:
            out = _mimicMat(self)
        else:
            out = Mat(self.shape, dtype=float64, UMat=self.UMat)
        cv.divide(src2, self._, dst=out._, dtype=CV_[out.dtype.name])
        return out
    def __itruediv__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        if not CV_[self.dtype.name] in [cv.CV_32F, cv.CV_64F]:
            raise TypeError("Cannot true divide integer in-place")
        cv.divide(self._, src2, dst=self._, dtype=CV_[self.dtype.name])
        return self

    def __floordiv__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        if CV_[self.dtype.name] in [cv.CV_32F, cv.CV_64F]:
            out = _mimicMatZeros(self)
            temp = Mat(self.shape, dtype=int32, UMat=self.UMat)
        else:
            out = _mimicMat(self)
            temp = out
        cv.divide(self._, src2, dst=temp._, dtype=CV_[temp.dtype.name])
        if not temp is out:
            cv.add(temp._, out._, dst=out._, dtype=CV_[self.dtype.name])
        return out
    def __rfloordiv__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        if CV_[self.dtype.name] in [cv.CV_32F, cv.CV_64F]:
            out = _mimicMatZeros(self)
            temp = Mat(self.shape, dtype=int32, UMat=self.UMat)
        else:
            out = _mimicMat(self)
            temp = out
        cv.divide(src2, self._, dst=temp._, dtype=CV_[temp.dtype.name])
        if not temp is out:
            cv.add(temp._, out._, dst=out._, dtype=CV_[self.dtype.name])
        return out
    def __ifloordiv__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        if CV_[self.dtype.name] in [cv.CV_32F, cv.CV_64F]:
            temp = Mat(self.shape, dtype=int32, UMat=self.UMat)
        else:
            temp = self
        cv.divide(self._, src2, dst=temp._, dtype=CV_[temp.dtype.name])
        if not temp is self:
            cv.add(temp._, 0, dst=self._, dtype=CV_[self.dtype.name])
        return self

    def __mod__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        out = self // value
        out *= value
        temp = self.copy()
        cv.add(
            temp._, src2, dst=temp._,
            mask=(temp<out)._, dtype=CV_[temp.dtype.name]
        )
        cv.subtract(temp._, out._, dst=out._)
        return out
    def __rmod__(self, value):
        out = value // self
        out *= self
        temp = _mimicMatZeros(self)
        temp += value
        cv.add(temp._, self._, dst=temp._, mask=(value<out)._)
        cv.subtract(temp._, out._, dst=out._)
        return out
    def __imod__(self, value):
        src2 = _src2(value, UMat=self.UMat)
        temp = self // value
        temp *= value
        cv.add(
            self._, src2, dst=self._,
            mask=(self<temp)._, dtype=CV_[self.dtype.name]
        )
        cv.subtract(self._, temp._, dst=self._)
        return self

    def __pow__(self, value, mod=None):
        out = _mimicMat(self)
        cv.pow(self._, value, dst=out._)
        if not mod is None:
            out %= mod
        return out
    def __ipow__(self, value):
        cv.pow(self._, value, dst=self._)
        return self

    def __matmul__(self, value):
        if not isinstance(value, Mat):
            raise TypeError("Must use Mat objects for @ matmul")
        elif not CV_[self.dtype.name] in [cv.CV_32F, cv.CV_64F]:
            raise TypeError("Must use float32 or float64 dtype for @ matmul")
        elif not self.dtype == value.dtype:
            raise TypeError("Dtypes must match for @ matmul")
        shape = (self.shape[0], value.shape[1])
        out = Mat(shape, dtype=self.dtype, UMat=self.UMat)
        cv.gemm(self._, value._, alpha=1, src3=None, beta=0, dst=out._)
        return out

    def __len__(self):
        return self.size
    def __getitem__(self, key):
        return self.n[key]
    def __setitem__(self, key, value):
        mat = self.n
        mat[key] = value
        if self.UMat:
            del self.u
            del self._
            self.u = cv.UMat(mat)
            self._ = self.u
        return value
    def __contains__(self, key):
        return key in self.n


def array(
    object, dtype=float64, copy=True, order='K', subok=False, ndmin=0,
    UMat=False, usageFlags=USAGE_DEFAULT
):
    mat = np.array(
        object, dtype=dtype, copy=copy, order=order, subok=subok, ndmin=ndmin
    )
    return Mat(mat.shape, mat=mat, UMat=UMat, usageFlags=usageFlags)


def zeros(
    shape, dtype=float64, order='C',
    UMat=False, usageFlags=USAGE_DEFAULT
):
    mat = np.zeros(shape, dtype=dtype, order=order)
    out = Mat(shape, mat=mat, UMat=UMat, usageFlags=usageFlags)
    return out


def ones(
    shape, dtype=float64, order='C',
    UMat=False, usageFlags=USAGE_DEFAULT
):
    out = zeros(
        shape, dtype=dtype, order=order, UMat=UMat, usageFlags=usageFlags
    )
    cv.add(out._, 1, dst=out._)
    return out


def nan_to_num(x, copy=True, nan=0.0):
    if copy:
        out = Mat(x.shape, dtype=x.dtype, UMat=x.UMat)
    else:
        out = x
    if CV_[out.dtype.name] == cv.CV_32F:
        cv.patchNaNs(out._, nan)
    elif CV_[out.dtype.name] == cv.CV_64F:
        mat = out.n
        np.nan_to_num(mat, copy=False, nan=nan)
        if out.UMat:
            del out.u
            del out._
            out.u = cv.UMat(mat)
            out._ = out.u
    return out
