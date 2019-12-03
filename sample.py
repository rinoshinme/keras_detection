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


def test_stack():
    image = np.random.rand(3, 5)
    print(image.shape)
    res1 = np.stack([image, image, image], axis=-1)
    res2 = np.concatenate([image, image, image], axis=-1)
    print(res1.shape)
    print(res2.shape)


if __name__ == '__main__':
    # test_argmax()
    test_stack()


