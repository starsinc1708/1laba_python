import numpy as np
import copy
import random

def REF(originalMatrix): #ступенчатый вид
    matrix = copy.deepcopy(originalMatrix)
    k = matrix.shape[0]
    n = matrix.shape[1]
    curRow = 0 #текущая ступенька
    for j in range(n): #проходим по всем столбцам
        for i in range(curRow + 1, k): #по строкам кроме уже построенных ступенек
            if matrix[i][j] == 1: #в этой строчке на данном столбце 1 быть не может
                if matrix[curRow][j] == 0: #если ступенька не построена
                    matrix[curRow] = (matrix[i] + matrix[curRow]) % 2 #строим ступеньку
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2 #обнуляем 1 в данной строчке
                else: #если уже построена
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2 #обнуляем 1 в данной строчке
        if matrix[curRow][j] == 1: #если в даном столбце построили новую ступеньку
            curRow += 1 #начинаем строить следующую ступеньку
            if curRow == k: #если все ступеньки построены, выходим
                break
    return matrix[0:curRow]

def RREF(originalMatrix): #приведённый ступенчатый вид
    matrix = copy.deepcopy(originalMatrix)
    k = matrix.shape[0]
    n = matrix.shape[1]
    curRow = 0 #текущая ступенька
    for j in range(n): #проходим по всем столбцам
        for i in range(curRow + 1, k): #по строкам кроме уже построенных ступенек
            if matrix[i][j] == 1: #в этой строчке на данном столбце 1 быть не может
                if matrix[curRow][j] == 0: #если ступенька не построена
                    matrix[curRow] = (matrix[i] + matrix[curRow]) % 2 #строим ступеньку
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2 #обнуляем 1 в данной строчке
                else: #если уже построена
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2 #обнуляем 1 в данной строчке
        if matrix[curRow][j] == 1: #если в даном столбце построили новую ступеньку
            for i in range(curRow): #по уже построенным ступенькам
                if matrix[i][j] == 1: #в этой строчке на данном столбце 1 быть не может
                    matrix[i] = (matrix[i] + matrix[curRow]) % 2 #складываем строки
            curRow += 1 #начинаем строить следующую ступеньку
            if curRow == k: #если все ступеньки построены, выходим
                break
    return matrix[0:curRow]

def introduceError(n, t):
    vector = np.zeros(n, int)
    errorPositions = random.sample(range(n), t)
    for i in range(t):
        vector[errorPositions[i]] = 1
    return vector

class LinearCode:

    def __init__(self,  S):
        self.S = S
        self.G = REF(self.S)
        self.k = self.G.shape[0]
        self.n = self.G.shape[1]
        self.RG = RREF(self.G)
        self.positions = self._getPositions(self.RG)
        self.X = self._getX(self.positions)
        self.H = self._getH(self.positions)
        self.C1 = self._getC1(self.G)
        self.C2 = self._getC2(self.G)
        self.d = self._getd(self.C1)
        self.t = self.d - 1
        self.Err = self._getErr(self.C1)

    def _getPositions(self, originalMatrix): #номера позиций ведущих столбцов
        pos = np.zeros((self.k), dtype = int)
        i = 0
        for j in range(self.n): #проходим по столбцам
            if i == self.k: #если все ведущие столбцы найдены
                break #выходим
            if originalMatrix[i][j] == 1: #если это ведущий столбец
                pos[i] = j #записываем его позицию
                i += 1 #выходим
        return pos

    def _getX(self, positions): #сокращенная матрица X
        X = np.zeros((self.k, self.n - self.k), dtype = int)
        p = 0
        for j in range(self.n): #проходим по столбцам
            if j - p == self.k: #если ведущие столбцы закончились
                X[:, p : self.n - self.k] = self.RG[:, j : self.n]
                break #копируем все оставшиеся столбцы и выходим
            if j != positions[j-p]: #если это не ведущий столбец
                X[:, p] = self.RG[:, j] #копируем его
                p += 1 #идём дальше
        return X

    def _getH(self, positions): #матрица H
        H = np.zeros((self.n, self.n - self.k), dtype = int) #заполняем нулями для начала
        p = 0
        for i in range(self.n): #проходим по строкам
            if p == self.k: #если всю матрицу X вписали
                for j in range(self.n-1, i - 1, -1):
                    H[j][j-k] = 1 #дабавляем единички в строки единичной матрицы
                break #выходим
            if i == positions[p]: #если номер строки соответствует номеру ведущего столбца
                H[i] = self.X[p] #записываем строчку из X
                p += 1 #идем дальше
            else:
                H[i][i-p] = 1 #ставим единичку в строку единичной матрицы
        return H

    def _getC1(self, G): #все кодовые слова первым способом
        C = np.zeros((pow(2, self.k), self.n), dtype = int) #сюда их запишем
        for j in range(self.k): #проходим по столбцам матрицы всевозможных сумм (по слагаемым)
            for p in range(pow(2, j)): #проходим по блокам из единиц в данном столбце
                for i in range(pow(2,self.k-j)*p+pow(2,self.k-j-1), pow(2,self.k-j)*(p+1)): #по единицам
                    C[i] = (C[i] + G[j]) % 2 #так как на данной позиции 1 - записываем это слагаемое в сумму

        C = [list(x) for x in set(tuple(x) for x in C)] #оставляем только уникальные строки
        C.sort(reverse=False) #сортируем, чтобы сравнить со вторым способом
        C = np.array(C) #возращаем к матрице
        return C

    def _getC2(self, G): #все кодовые слова вторым способом
        C = np.zeros((pow(2, self.k), self.n), dtype = int) #сюда их запишем
        D = np.zeros((pow(2, self.k), self.k), dtype = int) #матрица всех двоичных комбинаций
        for j in range(self.k):
            for p in range(pow(2, j)):
                for i in range(pow(2,self.k-j)*p+pow(2,self.k-j-1), pow(2,self.k-j)*(p+1)):
                    D[i][j] = 1 #ставим единички на соответствующие места
        for i in range(pow(2, self.k)):
            C[i] = D[i].dot(self.G) % 2 #каждую строчку на матрицу G
        C = [list(x) for x in set(tuple(x) for x in C)] #оставляем только уникальные строки
        C.sort(reverse=False) #сортируем, чтобы сравнить с первым способом
        C = np.array(C) #возращаем к матрице
        return C

    def _getd(self, C): #возвращает расстояние
        d = self.n #между словами длины n расстояние не больше n
        currSum = 0 #текущее расстояние
        for i in range(pow(2, self.k)-1): #проходим по всем строкам кроме последней
            for p in range(i+1, pow(2, self.k)): #сравниваем со всеми строками начиная со следующей
                for j in range(self.n): #проходим по этим двум словам
                    currSum += (C[i][j]+C[p][j])%2 #считаем расстояние
                if d > currSum: #выбираем минимальное расстояние
                    d = currSum
                currSum = 0
        return d

    def _getErr(self, C): #возвращает ошибку кратности t и слово, к которому ее надо применить
        pos = np.zeros((self.n+1), dtype = int) #в конце ошибки будет записан номер слова
        currPos = np.zeros((self.n+1), dtype = int)
        d = self.n #аналогично методу нахождения d, но теперь запоминаем еще и позиции
        currSum = 0
        for i in range(pow(2, self.k)-1):
            for p in range(i+1, pow(2, self.k)):
                for j in range(self.n):
                    currPos[j] = (C[i][j]+C[p][j])%2
                    currSum += currPos[j]
                if d > currSum:
                    d = currSum
                    for k in range(self.n):
                        pos[k] = currPos[k] #копируем ошибку
                    pos[self.n] = p
                currSum = 0
        return pos

k = 5
n = 10
matrix = np.random.randint(0, 2, (k, n)) #случайная матрица n*m из 0 и 1
print("original matrix S =")
print(matrix, end = "\n\n")
lc = LinearCode(matrix) #класс, который содержит линейный код и все его параметры
print("S =")
print(lc.S, end = "\n\n")
print("G =")
print(lc.G, end = "\n\n")
print("k =", lc.k, "  n =", lc.n, end = "\n\n")
print("G* =")
print(lc.RG, end = "\n\n")
#print(lc.positions, end = "\n\n")
print("X =")
print(lc.X, end = "\n\n")
print("H =")
print(lc.H, end = "\n\n")
print("C1 =")
print(lc.C1, end = "\n\n")
print("C2 =")
print(lc.C2, end = "\n\n")
print("check C1:")
for i in range(lc.C1.shape[0]):
    print(lc.C1[i].dot(lc.H)%2)
print("\n")
print("check C2:")
for i in range(lc.C2.shape[0]):
    print(lc.C2[i].dot(lc.H)%2)
print("\n")
print("d =", lc.d, "  t =", lc.t, end = "\n\n")
print("v =", lc.C1[lc.Err[lc.n]], end = "\n\n")
err1 = introduceError(lc.n, lc.t)
print("e1 =", err1)
print("v + e1 =", (lc.C1[lc.Err[lc.n]]+err1)%2)
print("(v + e1)@H =", ((lc.C1[lc.Err[lc.n]]+err1)%2).dot(lc.H)%2, end = "\n\n")
print("e2 =", lc.Err[:lc.n])
print("v + e2 =", (lc.C1[lc.Err[lc.n]]+lc.Err[:lc.n])%2)
print("(v + e2)@H =", ((lc.C1[lc.Err[lc.n]]+lc.Err[:lc.n])%2).dot(lc.H)%2)