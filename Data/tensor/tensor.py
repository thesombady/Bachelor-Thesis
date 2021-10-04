import numpy as np
import os
global hbar
import pandas as pd
hbar = 1.0545718 * 10 ** (-34)

"""We note, in the tensor-class below, we count double indexing, such that rho_{alpha, beta} -> rho[[alpha], [beta]] = rho[[j,m],[l,n]].
This indexing, is then row versus column. So 'alpha', decides row, and 'beta' decides column. """


class tensor():# A tensor class explicitly made for a three level maser

    def __init__(self, N = 3, Data = None):
        self.N = N # Number of photon modes
        try:
            if Data ==  None:
                self.Data = Data
        except:
            self.Data = Data

    def __repr__(self):
        return 'Data: {Data}\nPhoton modes : {number}'.format(Data = self.Data, number = self.N)

    def trace(self):
        """Find the trace of the density operator"""
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
        """
        try:
            alpha = index[0]
            beta = index[1]
        except Exception as E:
            raise(E)
        if alpha[1] == 1: # m
            alpha1 = 0
        elif alpha[1] == 2: # m
            alpha1 = self.N - 1
        elif alpha[1] == 3: #m
            alpha1 = 2 * self.N - 1
        if beta[1] == 1: #n
            beta1 = 0
        elif beta[1] == 2: #n
            beta1 = self.N - 1
        elif beta[1] == 3: #n
            beta1 = 2 * self.N - 1
        print((alpha1 + alpha[0] , beta1 + beta[0]))
        return self.Data[alpha1 + alpha[0], beta1 + beta[0]]
        """
        alpha = index[0]
        beta =  index[1]



    def _set(self, Val, index):
        """ Inserting the value Val, at index index, where index is a double index. such that [[j,m],[l,n]]"""
        try:
            alpha = index[0]
            beta = index[1]
        except Exception as E:
            raise (E)
        if alpha[1] == 1:  # m
            alpha1 = 0
        elif alpha[1] == 2:  # m
            alpha1 = self.N
        elif alpha[1] == 3:  # m
            alpha1 = 2 * self.N
        if beta[1] == 1:  # n
            beta1 = 0
        elif beta[1] == 2:  # n
            beta1 = self.N
        elif beta[1] == 3:  # n
            beta1 = 2 * self.N
        try:
            self.Data[alpha1 + alpha[0], beta1 + beta[0]] = Val
            #print(self.Data[int2 + index2[0], int1 + index1[0]])
        except Exception as E:
            raise(E)

    def _save(self, Name):
        Data = self.Data.copy()
        df = pd.DataFrame(Data)
        path = os.getcwd()
        path = os.path.join(path, '{name}.csv'.format(name = Name))
        df.to_csv(path)

    def __add__(self, other):
        Data = self.Data + other.Data /(self.Data.sum() + other.Data.sum())
        return tensor(N = self.N, Data = self.Data + other.Data)#Data

    def __radd__(self, other):
        Data = self.Data + other.Data / (self.Data.sum() + other.Data.sum())
        return tensor(N = self.N, Data = self.Data + other.Data)#)Data

    def __mul__(self, other):
        try:
            return tensor(N = self.N, Data = self.Data * other)
        except Exception as E:
            raise(E)

    def __rmul__(self, other):
        try:
            return tensor(N = self.N, Data = self.Data * other)
        except Exception as E:
            raise(E)

    def __truediv__(self, other):
        if isinstance(other, (float, int, complex)):
            if isinstance(other, complex):
                realdata = self.Data.real / other
                imdata = self.Data.imag / other
                return tensor(N = self.N, Data = realdata + imdata*1j)
            else:
                return tensor(N = self.N, Data = self.Data / other)
        else:
            pass

    def __rdiv__(self, other):
        if isinstance(other, (float, int, complex)):
            if isinstance(other, complex):
                realdata = self.Data.real / other
                imdata = self.Data.imag / other
                return tensor(N = self.N, Data = realdata + imdata*1j)
            else:
                return tensor(N = self.N, Data = self.Data / other)
        else:
            pass


class tens():

    def __init__(self, N = 3, Data = None):
        self.N = N
        try:
            if Data == None:
                self.Data = Data
        except:
            self.Data = Data


    def set1(self):
        """This is not used."""
        ones = np.ones(3 * self.N)
        return tensor(self.N, Data = np.matrix([ones / ((i ** 100 * 2 + 1)) * hbar for i in range(len(ones))], dtype = complex))

    def set0(self):
        """Set a 'tensor' with only zero elements"""
        zeros = np.zeros(3 * self.N)
        return tensor(self.N, Data = np.matrix([zeros for i in range(len(zeros))], dtype = complex))

    def zerostates(self):
        """Return a density operator, which only fills the zero-modes with quanta. Could be used as a initial-condition"""
        zeros = np.zeros(3 * self.N)
        tens = tensor(self.N, Data = np.matrix([zeros for i in range(len(zeros))], dtype = complex))
        value = hbar
        for j in range(self.N):
            for m in range(1,4):
                for l in range(self.N):
                    for n in range(1,4):
                        if m == 1 and n == 1:
                            tens._set(Val = 1, index = [[j,m], [l,n]])
        return tens
#tens = tensor0()
#tens = tens.set()
#print(tens)
#print(tens[[0,1],[0,2]])
