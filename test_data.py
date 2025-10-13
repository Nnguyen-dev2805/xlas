import numpy as np

TEST_MATRIX_1 = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ],
    dtype=np.uint8,
)

TEST_MATRIX_2 = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ],
    dtype=np.uint8,
)

TEST_MATRIX = {
    "gradient": np.array(
        [[i * 25 for i in range(10)] for _ in range(10)], dtype=np.uint8
    ),
    "checkerboard": np.array(
        [[255 if (i + j) % 2 == 0 else 0 for i in range(10)] for j in range(10)],
        dtype=np.uint8,
    ),
    'random': np.random.randint(0, 256, (10, 10), dtype=np.uint8)
}

def get_test_matrix_to_rgb(matrix):
    return np.stack([matrix]*3, axis=-1)


TEST_MATRIX_3 = np.array([[1,2,3,4,5,6,4,5,2,1,5],
                     [7,5,0,1,2,7,4,5,2,1,5],
                    [6,5,2,6,3,2,4,5,2,1,5],
                    [2,1,4,6,6,6,4,5,2,1,5],
                    [5,6,3,2,2,6,4,5,2,1,5],
                    [5,2,3,4,1,7,4,5,2,1,5],
                    [6,5,2,6,3,2,4,5,2,1,5],
                    [6,5,2,6,3,2,4,5,2,1,5]]).astype(np.uint8)

