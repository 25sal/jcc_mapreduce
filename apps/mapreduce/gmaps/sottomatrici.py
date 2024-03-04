import numpy as np

def distribute_submatrices(main_matrix, N, submatrix_size):
    submatrices_per_worker = (main_matrix.shape[0] // submatrix_size) * (main_matrix.shape[1] // submatrix_size) // N
    workers_submatrices = []

    for worker_id in range(N):
        submatrix_batch = []
        for _ in range(submatrices_per_worker):
            start_row = np.random.randint(0, main_matrix.shape[0] - submatrix_size + 1)
            start_col = np.random.randint(0, main_matrix.shape[1] - submatrix_size + 1)
            submatrix = main_matrix[start_row:start_row + submatrix_size, start_col:start_col + submatrix_size]
            submatrix_batch.append(submatrix)
        workers_submatrices.append(submatrix_batch)

    return workers_submatrices

