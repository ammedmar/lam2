import numpy as np
import itertools


def rref(matrix):
    """Row reduced echelon form of matrix with mod 2 coefficients

    Parameters
    ==========

    matrix :
        2-d numpy array with boolean type

    Returns
    =======

    a pair of numpy arrays R, U satisfying as matrices M = UR where M is
    the original matrix R is in row echelon form and U is upper triangular
    and invertible.

    Examples
    ========

    >>> M = np.array([[0, 0, 0], \
                      [0, 1, 0], \
                      [1, 1, 0]], dtype=np.bool)

    >>> reduced, triangular, pivots = rref(M)

    >>> reduced.astype(np.int)
    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 0]])

    >>> triangular.astype(np.int)
    array([[0, 1, 1],
           [0, 1, 0],
           [1, 0, 0]])

    >>> pivots
    [[0, 1], [0, 1]]
    """

    pivots = [[], []]  # col-row
    lead = 0
    rowCount, columnCount = matrix.shape
    triangular = np.eye(rowCount, dtype=np.bool)
    for r in range(rowCount):
        if lead >= columnCount:
            break
        if np.any(matrix[r:, lead]):
            i = min(filter(lambda j: matrix[j, lead], range(r, rowCount)))
            matrix[[i, r], :] = matrix[[r, i], :]
            triangular[[i, r], :] = triangular[[r, i], :]
            indicesExcept = itertools.chain(range(0, r), range(r + 1, rowCount))
            for j in filter(lambda j: matrix[j, lead], indicesExcept):
                matrix[j, :] = np.logical_xor(matrix[r, :],
                                              matrix[j, :])
                triangular[j, :] = np.logical_xor(triangular[r, :],
                                                  triangular[j, :])
            pivots[0].append(r)
            pivots[1].append(lead)

        lead += 1

    return matrix, triangular, pivots


def solve(vector, matrix, reduced=None, triangular=None, pivots=None):
    '''Solves the linear system Mx=v

    >>> M = np.array([[1, 1, 1], \
                      [0, 1, 0], \
                      [0, 0, 0]])
    >>> v = np.array([0, 0, 1])
    >>> solve(v, M)
    Traceback (most recent call last):
    ValueError: equation has no solutions

    '''
    m, n = matrix.shape
    if reduced is None:
        reduced, triangular, pivots = rref(matrix)
    # Apply row operations to input vector
    w = np.matmul(triangular.astype(np.int8), vector.astype(np.int8))
    w = (w % 2).astype(np.bool)
    # Solvable?
    nonZeroRows = [i for i, b in enumerate(w) if b]
    if not all(i in pivots[0] for i in nonZeroRows):
        raise ValueError('equation has no solutions')
    # Find one solution
    nonZeroColumns = [col for row, col in zip(*pivots) if w[row]]
    solution = np.zeros(n, dtype=np.bool)
    for col in nonZeroColumns:
        solution[col] = True
    # Solutions to homogeneous system
    freeColumns = [i for i in range(n) if i not in pivots[1]]
    kernel_basis = np.zeros((n, len(freeColumns)), dtype=np.bool)
    for idx, i in enumerate(freeColumns):
        freeColumn = reduced[:, i]
        for row in filter(lambda row: freeColumn[row], range(m)):
            col = pivots[1][pivots[0].index(row)]
            kernel_basis[col, idx] = True
        kernel_basis[i, idx] = True

    return solution, kernel_basis


if __name__ == "__main__":
    import doctest
    doctest.testmod()
