import numpy as np


class tensor:

    def __init__(self, data, N = 3):
        """Returns a tensor for a three level maser with N numbers of photon modes.
        Is used to retrieve the density operators values for the different modes."""
        try:
            if data.any() == None:
                self.data = None
            else:
                self.data = data
            self.N = N
        except:
            print("didn't work")

    def __add__(self, other):
        if isinstance(other, tensor):
            return tensor(self.data + other.data, self.N)
        else:
            raise TypeError("Cannot add since of different format")


    def __radd__(self, other):
        if isinstance(other, tensor):
            return tensor(self.data + other.data, self.N)
        else:
            raise TypeError("Cannot add since of different format")


    def __getitem__(self, index):
        """Indicies are a tuple of (i,m) where i is the photon number mode and m is the Hilbert quantum state.
        Indices range from 0 to 299, where (0,0) = 100, 101 = (1,0), and (99,3) = 499."""
        if not isinstance(index, (tuple, list, np.generic, np.ndarray)):
            raise TypeError("Indicies of different format")
        else:
            index1 = index[0][1]  + index[0][0]
            index2 = index[1][1]  + index[1][0]
            print("index1", index1)
            print("index2", index2)

            """
            index1 = index1[0][1] + 100 + index1[0][0]
            index2 = index1[1][1] + 100 + index1[1][0]
            return self.data[index1][index2]
            """

    def __repr__(self):
        return f'{self.data}'


    def _set(self, data):
        if not isinstance(data, (np.generic, np.ndarray, np.matrix)):
            raise TypeError("Data has to be of mutable type")
        else:
            N = self.N
            indicies = np.array([m + 100 + j for j in range(N) for m in range(3)])
            """ Indicies range from 100, to 400 + N, where 100 would be (0,0), 101 would be (1,0)
            and (3,99), would be 499 since N is maximum 99 in this computation.
            """
            if len(data) != (3 * N) ** 2:
                raise TypeError("Mismatch of order")
            else:
                self.data = np.matrix(self.data)

class tensor0(tensor):

    def __init__(self, data = None, N = 3):
        self.data = data
        self.N = N


    def set(self):
        zeros = np.ones(3 * self.N)
        return tensor(np.matrix([zeros * i for i in range(len(zeros))]), self.N)


x1 = tensor0(N = 3).set()
print(x1[(0,0), (3,0)])
print(x1)
