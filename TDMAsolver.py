import numpy as np
import typing


class TDMAsolver:
    """
    Implements tridiagonal matrix algorithm.
    The designations are the same as in the book Samarsky, Nikolaev.

    Usage:

        1) Create matrices A,B,C, and F
        2) Create an object of the TDMAsolver class, like 
           solver  = TDMAsolver(N,d,dtype = someDataType)   (N - number of grid nodes,
           d - number of equations in original ODE system)
        3) Pass matrices into it using solver.# = someNumpyArray (# = A,B,C,F)
        4) Calculate the solution by calling the solver.solve()
        5) Solution is accessible with solver.solution

    """

    __N: int
    """
        Number of grid nodes
    """

    __dim: int
    """
        Number of ODEs in original problem
    """

    __A: np.ndarray
    __B: np.ndarray
    __C: np.ndarray
    __F: np.ndarray

    __solution: np.ndarray
    __alpha: np.ndarray
    __beta: np.ndarray

    __dtype: np.dtype

    __WRONG_MATRIX_SIZE: str = "Wrong matrix size."

    def __init__(self, N: int, dim: int = 1, dtype: np.dtype = np.double) -> None:
        self.__dim = dim
        self.__N = N
        self.__dtype = dtype

        if dim == 1:
            self.__solution = np.zeros(N, dtype=dtype)
            self.__beta = np.zeros(N, dtype=dtype)
            self.__alpha = np.zeros(N, dtype=dtype)
        else:
            self.__solution = np.zeros((N, dim), dtype=dtype)
            self.__beta = np.zeros((N, dim), dtype=dtype)
            self.__alpha = np.zeros((N, dim, dim), dtype=dtype)

    def _checkMatrix1D(self, shape: typing.Tuple[int]) -> None:
        if self.__dim != 1 or self.__N != shape[0]:
            raise RuntimeError(self.__WRONG_MATRIX_SIZE)
        return

    def _checkMatrixABC(self, mat: np.ndarray) -> None:
        shape: typing.Tuple[int] = mat.shape
        if len(shape) == 1:
            self._checkMatrix1D(shape)
            return
        if len(shape) == 3:
            if self.__dim != shape[2] or self.__dim != shape[1] or self.__N != shape[0]:
                raise RuntimeError(self.__WRONG_MATRIX_SIZE)
            return
        raise RuntimeError(self.__WRONG_MATRIX_SIZE)

    def _checkMatrixF(self, mat: np.ndarray) -> None:
        shape: typing.Tuple[int] = mat.shape
        if len(shape) == 1:
            self._checkMatrix1D(shape)
            return
        if len(shape) == 2:
            if self.__dim != shape[1] or self.__N != shape[0]:
                raise RuntimeError(self.__WRONG_MATRIX_SIZE)
            return
        raise RuntimeError(self.__WRONG_MATRIX_SIZE)

    def _solve1D(self) -> None:
        self.__alpha[0] = self.__B[0] / self.__C[0]
        self.__beta[0] = self.__F[0] / self.__C[0]

        for i in range(1, self.__N):
            denumenator = self.__C[i] - self.__alpha[i - 1] * self.__A[i]
            self.__alpha[i] = self.__B[i] / denumenator
            self.__beta[i] = (
                self.__F[i] + self.__A[i] * self.__beta[i - 1]
            ) / denumenator

        self.__solution[-1] = self.__beta[-1]

        for i in range(self.__N - 1, 0, -1):
            self.__solution[i - 1] = (
                self.__alpha[i - 1] * self.__solution[i] + self.__beta[i - 1]
            )

    def _solveND(self) -> None:
        denumenator = np.linalg.inv(self.__C[0])
        self.__alpha[0] = np.dot(denumenator, self.__B[0])
        self.__beta[0] = np.dot(denumenator, self.__F[0])

        for i in range(1, self.__N):
            denumenator = np.linalg.inv(
                self.__C[i] - np.dot(self.__A[i], self.__alpha[i - 1])
            )
            self.__alpha[i] = np.dot(denumenator, self.__B[i])
            self.__beta[i] = np.dot(
                denumenator, self.__F[i] + np.dot(self.__A[i], self.__beta[i - 1])
            )
        self.__solution[-1] = self.__beta[-1]

        for i in range(self.__N - 1, 0, -1):
            self.__solution[i - 1] = (
                np.dot(self.__alpha[i - 1], self.__solution[i]) + self.__beta[i - 1]
            )

    @property
    def A(self):
        return self.__A

    @A.setter
    def A(self, A: np.ndarray):
        self._checkMatrixABC(A)
        self.__A = A

    @property
    def B(self):
        return self.__B

    @B.setter
    def B(self, B: np.ndarray):
        self._checkMatrixABC(B)
        self.__B = B

    @property
    def C(self):
        return self.__C

    @C.setter
    def C(self, C: np.ndarray):
        self._checkMatrixABC(C)
        self.__C = C

    @property
    def F(self):
        return self.__F

    @F.setter
    def F(self, F: np.ndarray):
        self._checkMatrixF(F)
        self.__F = F

    def solve(self) -> None:
        if self.__dim == 1:
            self._solve1D()
        else:
            self._solveND()

    @property
    def solution(self):
        return self.__solution


# For tests:
if __name__ == "__main__":
    pass
    # import matplotlib.pyplot as plt
    # N = 300
    # dim = 2
    # h = 1/(N-1)
    # a = np.zeros((N, dim, dim))
    # b = np.zeros_like(a)
    # c = np.zeros_like(a)
    # f = np.zeros((N, dim))
    # for i in range(1, N-1):
    #     a[i][1][1] = 1
    #     b[i][1][1] = 1
    #     c[i][0][0] = 1
    #     c[i][1][1] = 2
    #     c[i][1][0] = h**2
    #     f[i][0] = 1
    # c[0][0][0] = 1
    # c[0][1][1] = 1
    # f[0][0] = 1
    # c[-1][0][0] = 1
    # c[-1][1][1] = 1
    # f[-1][0] = 1
    # f[-1][1] = 1

    # solver = TDMAsolver(N, dim)

    # solver.setMatrixA(a)
    # solver.setMatrixB(b)
    # solver.setMatrixC(c)
    # solver.setMatrixF(f)

    # solver.solve()
    # solution = solver.getSolution()
    # plt.figure(1)
    # plt.plot(np.linspace(0, 1, N), solution[:,0],'b')
    # plt.plot(np.linspace(0, 1, N), solution[:,1],'r--')
    # plt.show()
