import numpy as np
import pandas as pd



class tensor():
	
	def __init__(self, N = 3):
		"""A simple tensor class used for a three-level maser with N particles.
The majority of the computations has been done by hand, and the program, calc.py,
utalizes those calculations. """
		self.N = N

	def _set0(self):
		Number = self.N * 3
		zeros = np.zeros(Number)
		matrix1 = np.matrix([zeros for i in range(len(zeros))])
		matrix2 = np.matrix([zeros for i in range(len(zeros))])
		self.Data = [matrix1, matrix2]	
	
	def _set1(self):
		Number = self.N * 3
		ones = np.ones(Number)
		matrix1 = np.matrix([ones for i in range(len(ones))])
		matrix2 = np.matrix([ones for i in range(len(ones))])
		self.Data = [matrix1, matrix2]	


	def __repr__(self):
		try:
			return f'{self.Data[0}\n{self.Data[1]}\nPhotonmodes {self.N}'
		except Exception as E:
			raise E
	
	

	def __getitem__(self, index):
		"""index is a two dimensional array with each element being a tuple,
		 list or array of dimension two. """
		index1 = index[0]; index2 = index[1]
		state1 = index1[1]; state2 = index2[1]
		photon1 = index1[0]; photon2 = index2[0]
				



