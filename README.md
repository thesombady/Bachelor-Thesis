# My Github
![Your Repository’s Stats](https://github-readme-stats.vercel.app/api?username=thesombady&show_icons=true)

![Your Repository's Stats](https://github-readme-stats.vercel.app/api/top-langs/?thesombady=thesombady&theme=blue-green)

<img src=https://komarev.com/ghpvc/?username = thesombady/>

Above you can see a summary of my activity on Github.
## My journey
I have been working on loads of small projects corresponding to course-work and self-interest in python, which can be viewed in my public repositories.
A few of my larger projects include: Newton Fractals, Sacra, and Bachelor-Thesis, were one is a course-work, Newton Fractals, one is of own self-interest, Sacra, and lastly the Bachelor-Thesis project is that which means to examinate my learnings in the bachelor program of physiscs at Lund University.

# Bachelor Thesis
Bachelor-thesis project for Andreas Evensen at Lund University. This project involves simulations for a three-level maser, built upon the quantum treatment of electromagnetic field and light-fields.


# Problem
The program is used to solve, by the Runge-Kutta method, the following equation,

<img src="https://render.githubusercontent.com/render/math?math=\dot{\rho}=\frac{1}{i\hbar}\Big[\hat{H},\hat{\rho}\Big]\pm\sum_{i\in\{c,h\}}\mathcal{L}_i[\hat{\rho}]">

Note, there is no minus, could just not render without it. Furthermore, the density matrix is of pure-states, such that the standards relationships holds. Therefore, it's easy to extract
the entropy of the system, which can be related to the state of the system evolution.
## Outline 
The problem is solved in terms of compressed indices, where then the density operator, rho, is a square matrix. Hence, each element of the density operator/matrix, is computed and then retransformed into a operator.
Therefore, the program itself is slow, simply because it's not possible to do concurrent calculations with iterative method.

Moreover, the program itself has specific description. If used, read the description to fully understand what each element does.


### Disclaimer
This work is done on request of the thesis supervisor. This program will be updating until the end of the simulation period, which may be the entire thesis duration!
