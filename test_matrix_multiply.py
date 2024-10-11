import numpy as np
import pytest
from matrix_operations import track_memory, track_cpu,matrix_multiply

@pytest.fixture
def setup_matrices():
    size = 1024
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.zeros((A.shape[0],B.shape[1]),dtype = float)
    blockSize = 64
    return A, B, C, size, blockSize

@pytest.mark.benchmark(min_rounds=1)
def test_matrix_multiply(benchmark, setup_matrices):
    A, B , C, N, blockSize= setup_matrices

    result = benchmark(matrix_multiply, A, B, C, N, blockSize)

    track_memory(matrix_multiply, A, B, C, N, blockSize)
    track_cpu(matrix_multiply, A, B, C, N, blockSize)

    assert result is not None
