import copy
import numpy as np


def swap(row1, row2, len):
    for i in range(len):
        row1[i], row2[i] = row2[i], row1[i]
    return row1, row2


# 1.1. Реализовать функцию REF(), приводящую матрицу к ступенчатому виду.
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
    return matrix[0:curRow]


# 1.2. Реализовать функцию RREF(), приводящую матрицу к приведённому ступенчатому виду.
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
    return matrix[0:curRow]


class LinearCode:

    def __init__(self, M):
        self.M = M
        self.G = REF(self.M)
        self.RG = RREF(self.G)
        self.k = self.G.shape[0]
        self.n = self.G.shape[1]
        self.lead = self.getPositions(self.RG)
        self.X = self.createX(self.lead, self.RG)
        self.H = self.createH(self.lead, self.X)
        self.Code1 = self.createCode1(self.G)

    # ведущие столбцы lead матрицы 𝐆∗
    def getPositions(self, originalMatrix):
        lead = np.zeros(self.k, dtype=int)
        j = 0
        for i in range(self.n):  # проходим по столбцам
            if j == self.k:  # если все ведущие элементы найдены
                break
            if originalMatrix[j][i] == 1:  # если это ведущий столбец
                lead[j] = i  # записываем позицию ведущего элемента
                j += 1
        return lead

    # сокращённая матрица 𝐗, удалив ведущие столбцы матрицы G∗
    def createX(self, lead, originalMatrix):
        new_M = np.zeros((self.k, self.n - len(lead)), dtype=int)
        n = 0
        for i in range(self.k):
            for j in range(self.n):
                if j not in lead:
                    new_M[i][n] = originalMatrix[i][j]
                    n += 1
            n = 0
        return new_M

    # матрица 𝐇, строки,соответствующие позициям ведущих столбцов строки из 𝐗,
    # а остальные – строки единичной матрицы.
    def createH(self, lead, X):
        H = np.zeros((self.n, self.n - self.k), dtype=int)
        t = 0
        j = 0
        for i in range(self.n):
            if i in lead:
                H[i] = X[t]
                t += 1
            else:
                H[i][j] = 1
                j += 1
        return H

    def createCode1(self, G):
        sum = 0
        for i in range(self.k + 1):
            sum += i
        Words = np.zeros((sum, self.n), dtype=int)
        t = 0
        for i in range(self.k):
            for j in range(i, self.k):
                Words[t] = (G[i] + G[j]) % 2
                t += 1
        Words = np.unique(Words, axis=0)
        return Words


# 1.3.1 На основе входной матрицы сформировать порождающую матрицу в ступенчатом виде
matrix = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
                   [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
                   [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                   [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
                   [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=int)

lc = LinearCode(matrix)

print("S:\n", matrix)
print("S_REF:\n", REF(matrix))
new_matrix = matrix
print("S_RREF:\n", RREF(new_matrix))

# 1.3.2 Задать n равное числу столбцов и k равное числу строк полученной матрицы (без учёта полностью нулевых строк)
G = REF(matrix)
print("n = ", G.shape[1])
print("k = ", G.shape[0])
print("------------------")

# 1.3.3 Сформировать проверочную матрицу на основе порождающей
print("G:\n", G)
G = RREF(G)
print("G*:\n", G)
print("lead = ", lc.lead)
print("X:\n", lc.X)
print("H:\n", lc.H)
print("---------------------------")

# 1.4. Сформировать все кодовые слова длины 11 двумя способами.
# 1.4.1 Сложить все слова из порождающего множества, оставить
# неповторяющиеся.

# 1.4.2 Взять все двоичные слова длины 5, умножить каждое на G.
G = np.array([[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
              [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
              [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=int)

print("G:\n", lc.G)
print("H:\n", lc.H)
print("Result:")
print("C1:\n", lc.Code1)
