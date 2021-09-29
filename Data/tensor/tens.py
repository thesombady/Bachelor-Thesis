import numpy as np
import pandas as pd




class tensor():# A tensor class explicitly made for a three level maser
    
    def __init__(self, N = 3, Data = None):
        self.N = N;# Number of photon modes
        try:
            if Data ==  None:
                self.Data = Data
        except:
            self.Data = Data


    def __repr__(self):
        return f'Data: {self.Data}\nPhoton modes : {self.N}'

    def _set(self, data1, data2, data3):
        self.matrix1 = np.matrix(data1)
        self.matrix2 = np.matrix(data2)
        self.Data = [self.matrix1, self.matrix2]

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
        Boundary = self.N * 3
        Data = self.Data.copy()
        state1 = index1[1]; state2 = index2[1]
        photonmode1 = index1[0]; photonmode2 = index2[0]
        int1 = photonmode1 + state1 ;  int2 = photonmode2 + state2
        return Data[int1, int2]
        



class tensor0(tensor):

    def __init__(self, N = 3, Data = None):
        self.N = N
        try:
            if Data == None:
                self.Data = Data
        except:
            self.Data = Data


    def set(self):
        ones = np.ones(3 * self.N)
        return tensor(N = 3, Data = np.matrix([ones * i for i in range(len(ones))]))

tens = tensor0()
tens = tens.set()
print(tens)
print(tens[[0,1],[0,2]])
