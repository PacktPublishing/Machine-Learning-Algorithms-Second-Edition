from __future__ import print_function

import numpy as np
# For further information, please visit https://cupy.chainer.org/
import cupy as cp
import time


# For reproducibility
np.random.seed(1000)
cp.random.seed(1000)

size = 5000


if __name__ == '__main__':
    # Create the matrices using NumPy
    A1 = np.random.normal(0.0, 2.0, size=(size, size)).astype(np.float32)
    A2 = np.random.normal(0.0, 2.0, size=(size, size)).astype(np.float32)

    # Perform the measurement using NumPy
    Ad = A1.copy()

    start_time = time.time()

    for _ in range(100):
        Ad = np.dot(Ad, A2)

    end_time = time.time()
    elapsed = end_time - start_time
    print(elapsed)

    # Create the matrices using CuPy
    B1 = cp.random.normal(0.0, 2.0, size=(size, size))
    B2 = cp.random.normal(0.0, 2.0, size=(size, size))

    # Perform the measurement using CuPy with GPU support
    Bd = B1.copy()

    start_time = time.time()

    for _ in range(100):
        Bd = cp.dot(Bd, B2)

    end_time = time.time()
    elapsed = end_time - start_time
    print(elapsed)



