"""
do random testing
"""

import numpy as np


def test_argmax():
    matrix = np.random.rand(5, 4)
    print(matrix.shape)
    matrix[1, :] = 0
    print(matrix)
    indices = np.argmax(matrix, axis=1)
    print(indices)



if __name__ == '__main__':
    test_argmax()



