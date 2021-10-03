# Bachelor Thesis
Bachelor-thesis project for Andreas Evensen at Lund University. This project involves simulations for a three-level maser, built upon the quantum treatment of electromagnetic field and light-fields.

# Problem
The program is used to solve, by the Runge-Kutta method, the following eqauation,
  <img src="https://render.githubusercontent.com/render/math?math=\dot{\rho}=\frac{1}{i\hbar}\Big[\hat{H},\hat{\rho}\Big]\pm\mathcal{L}_h\hat{\rho}\pm\mathcal{L}_c\hat{\rho}."> 

Note, there is no minus, could just not render without it. 
## Outline 
The problem is solved in terms of compressed indicies, where then the density operator, rho, is a square matrix. Hence, each element of the density operator/matrix, is computed and then retransformed into a operator.
Therefore, the program itself is slow, simply because it's not possible to do concurrent calculations with iterative method.

Moreover, the program itself has specific description. If used, read the description to fully understand what each element does.


### Disclaimer
This work is done on request of the thesis supervisor.
