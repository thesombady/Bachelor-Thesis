import numpy as np
import pandas as pd

class tensor():# A tensor class explicitly made for a three level maser

    def __init__(self, N = 3, Data = None):
        self.N = N # Number of photon modes
        try:
            if Data ==  None:
                self.Data = Data
        except:
            self.Data = Data


    def __repr__(self):
        return f'Data: {self.Data}\nPhoton modes : {self.N}'

    def trace(self):
        Number = 3 * self.N
        Trace = 0
        for i in range(Number):
            for j in range(Number):
                try:
                    if i == j:
                        Trace += self.Data[i, j]
                    else:
                        pass
                except Exception as E:
                    print(E)
        return Trace

    def __getitem__(self, index):
        """Getting the element beloning to alpha, beta.
        The alpha however is a double index from the ket [j,m>, where j is the photon mode, and m is the Hilbert-
        space basis, ranging from 0-2 or 1-3. The photon mode itself can go from 0, 99.
        Hence, index is a double-double index, such that [[j,m],[l,n]]. First index is the column, second the row
        Note that the elements in the matrix goes from (0,0), ..., (N,0), (0,1), ..., (N,1), (0,2), ..., (N, 2)"""
        try:
            index1 = index[0]
            index2 = index[1]
        except Exception as E:
            raise(E)
        int1 = 0
        int2 = 0
        if index1[1] == 1: # m
            int1 = 0
        elif index1[1] == 2: # m
            int1 = self.N - 1
        elif index1[1] == 3: #m
            int1 = 2 * self.N - 1
        if index2[1] == 1: #n
            int2 = 0
        elif index2[1] == 2: #n
            int2 = self.N - 1
        elif index2[1] == 3: #n
            int2 = 2 * self.N - 1
        return self.Data[int1 + index1[0], int2 + index2[0]]


    def _set(self, Val, index):
        """ Inserting the value Val, at index index, where index is a double index."""
        try:
            index1 = index[0]
            index2 = index[1]
        except Exception as E:
            raise (E)
        int1 = 0
        int2 = 0
        if index1[1] == 1:  # m
            int1 = 0
        elif index1[1] == 2:  # m
            int1 = self.N
        elif index1[1] == 3:  # m
            int1 = 2 * self.N
        if index2[1] == 1:  # n
            int2 = 0
        elif index2[1] == 2:  # n
            int2 = self.N
        elif index2[1] == 3:  # n
            int2 = 2 * self.N
        try:
            self.Data[int1 + index1[0], int2 + index2[0]] = Val
            #print(self.Data[int2 + index2[0], int1 + index1[0]])
        except Excpetion as E:
            raise(E)


class tens():

    def __init__(self, N = 3, Data = None):
        self.N = N
        try:
            if Data == None:
                self.Data = Data
        except:
            self.Data = Data


    def set1(self):
        ones = np.ones(3 * self.N)
        return tensor(self.N, Data = np.matrix([ones * i for i in range(len(ones))], dtype = complex))

    def set0(self):
        zeros = np.zeros(3 * self.N)
        return tensor(self.N, Data = np.matrix([zeros for i in range(len(zeros))], dtype = complex))


#tens = tensor0()
#tens = tens.set()
#print(tens)
#print(tens[[0,1],[0,2]])
