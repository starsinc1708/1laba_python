import copy

import numpy as np


def swap(row1, row2, len):
    for i in range(len):
        row1[i], row2[i] = row2[i], row1[i]

    return row1, row2

def RREF(origMatrix):
    matrix = copy.deepcopy(origMatrix)
    k = matrix.shape[0]
    n = matrix.shape[1]
    curRow = 0
    for j in range(n):
        for i in range(curRow + 1, k):
            if matrix[i][j] == 1:
                if matrix[curRow][j] == 0:
                    swap(matrix[curRow], matrix[i], k)
                else:
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2
        if matrix[curRow][j] == 1:
            curRow += 1
            if curRow == k:
                break

    for i in range(k - 1, -1, -1):
        curCol = 0
        for j in range(0, n):
            if matrix[i][j] == 1:
                curCol = j
                break
        for j in range(i):
            if matrix[j][curCol] == 1:
                matrix[j] = (matrix[j] + matrix[i]) % 2
    print(matrix)



def REF(origMatrix):
    matrix = copy.deepcopy(origMatrix)
    k = matrix.shape[0]
    n = matrix.shape[1]
    curRow = 0
    for j in range(n):
        for i in range(curRow + 1, k):
            if matrix[i][j] == 1:
                if matrix[curRow][j] == 0:
                    swap(matrix[curRow], matrix[i], k)
                else:
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2
        if matrix[curRow][j] == 1:
            curRow += 1
            if curRow == k:
                break
    print(matrix[0:curRow])



k = 6
n = 6
matrix = np.random.randint(0, 2, (k, n), dtype=int)
# matrix = np.array([[0, 1, 1, 0, 1, 1, 0, 0, 1, 0],
#   [0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
#   [0, 1, 0, 0, 0, 0, 1, 1, 0, 0],
#   [0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
#   [1, 1, 0, 1, 1, 0, 1, 0, 0, 0],
#   [0, 1, 1, 1, 0, 1, 1, 1, 1, 1]], dtype=int)
print(matrix)
REF(matrix)
RREF(matrix)
