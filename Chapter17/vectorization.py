from __future__ import print_function

import numpy as np
import time


# For reproducibility
np.random.seed(1000)


size = 500


if __name__ == '__main__':
    # Create the matrices
    A1 = np.random.normal(0.0, 2.0, size=(size, size)).astype(np.float32)
    A2 = np.random.normal(0.0, 2.0, size=(size, size)).astype(np.float32)

    # Non-vectorized computation
    D = np.zeros(shape=(size, size)).astype(np.float32)

    start_time = time.time()

    for i in range(size):
        for j in range(size):
            d = 0.0
            for k in range(size):
                d += A1[i, k] * A2[k, j]
            D[i, j] = d

    end_time = time.time()
    elapsed = end_time - start_time
    print(elapsed)

    # Vectorized computation
    start_time = time.time()

    D = np.dot(A1, A2)

    end_time = time.time()
    elapsed = end_time - start_time
    print(elapsed)