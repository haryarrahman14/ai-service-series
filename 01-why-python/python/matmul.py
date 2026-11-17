# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "numpy==2.5.5",
# ]
# ///
# AI series, part 1: pure Python vs NumPy on the same matrix multiplication.
import random
import time

import numpy as np

N = 200


def matmul_python(a, b):
    n, m, p = len(a), len(b), len(b[0])
    out = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            s = 0.0
            for k in range(m):
                s += a[i][k] * b[k][j]
            out[i][j] = s
    return out


def main():
    random.seed(0)
    a = [[random.random() for _ in range(N)] for _ in range(N)]
    b = [[random.random() for _ in range(N)] for _ in range(N)]

    t0 = time.perf_counter()
    c_py = matmul_python(a, b)
    py_ms = (time.perf_counter() - t0) * 1000

    a_np, b_np = np.array(a), np.array(b)
    np_ms = float("inf")
    for _ in range(20):
        t0 = time.perf_counter()
        c_np = a_np @ b_np
        np_ms = min(np_ms, (time.perf_counter() - t0) * 1000)

    assert np.allclose(c_py, c_np)
    print(f"Pure Python vs NumPy, {N}x{N} matrix multiplication")
    print(f"    pure Python  : {py_ms:.3f} ms")
    print(f"    NumPy (@)    : {np_ms:.3f} ms (best of 20)")
    print(f"    NumPy is {py_ms / np_ms:.0f}x faster")


if __name__ == "__main__":
    main()
