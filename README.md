# CM-2Norm_matrix_problem.
Final project for the course "Computational Mathematics for Learning and Data Analysis"  at university of Pisa for the A.Y 2020/2021. In this project we explore the problem of estimating the matrix 2-norm as an uncostrained optimization problem. We used methods "Conjugate Gradient Descent", "Quasi-Newton Method(BFGS)" and "Arnoldi process". All the theoretical analysis of these methods is included in the report.

# How to use
Each method can be tested on a jupyter notebook that is placed inside the folder.

# Guide CGD
Start by importing necessary functions and classes.
```python
# import
from cgd_funcs import Funcs
from cgd import CGD
```

Import or generate your matrix and initial vector.

```python
M1 = np.loadtxt('../data/M4.txt')
x0_m1 = np.loadtxt('../data/x0_m4.txt')
```

Initialize function and and cgd

```python
funcs = Funcs(M1)
# Initialize the CGD algorithm.
cgd = CGD(funcs.func_,funcs.func_grad_,funcs.exact_line_search,x0_m1, 1e-5, 1000, method='FR', verboose=True)
```
Possible parameters for method parameter for beta
```
FR for Fletcher-Reeves method
PR for Polak–Ribière method
HS for hessian-free method
```
Finally start the algorithm
```python
# Run the algorithm.
residual, errors, result = cgd.cgd()
```
