'''
  Authors: Dawit Anelay
           Marco Pitex
           Yohannis Telila
           
    Date : 24 / 11 / 2021 

  Implemetation of the CGD algorithm to solve the problem of estimating
  the matrix 2-norm as an uncostrained optimization problem.

'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from cgd_funcs import Funcs
from plots import plot
class CGD:

    # Initialize the CGD algorithm
    def __init__(self, func_, func_grad_,exact_line_search, x0, tol, max_iter, method='FR', verboose=False):
        '''
        Parameters
        ----------
            func_ : function, function to be minimized. 
                    f(x) = (x^T M x) / x^T x , where M = A^T A
        func_grad : function, Gradient of the function to be minimized. 
                    f'(x) = ((2 x (x^T M x)) / (x^T x)^2) - ((2 M x) / (x^T x))
               x0 : ndarray, initial guess. Can be set to be any numpy vector
           method : str, optional, Method to be used to calculate beta. can be on of the fol FR, PR, HS, DY, HZ.
                   "FR" for Fletcher-Reeves method,
                   "PR" for  ---- "PR" -----,  
                   "HS" for hessian-free method,
                   "DY" for Dykstra's method,
         max_iter : int , maximum number of iterations.
          verbose : bool, optional, if True prints the progress of the algorithm.

        '''
        self.func_ = func_
        self.func_grad_ = func_grad_
        self.exact_line_search = exact_line_search
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.verboose = verboose
    

    # CGD algorithm
    def cgd(self):
        '''
        
        Non linear CGD algorithm to solve the problem of estimating the matrix 2-norm as an
        unconstrained optimization.
        
        '''
        # initialize the variables.
        x = self.x0
        fx = self.func_(x)
        gfx = self.func_grad_(x)
        p = -gfx
        gfx_norm = np.linalg.norm(gfx)

        # Collect the results.
        num_iter = 0
        curve_x= [x]
        curve_fx = [fx]
        errors = []
        if(self.verboose):
            print('Initial condition: fx = {:.4f}, x = {} \n'.format(fx, x))

        # Start the algorithm iterations.
        while gfx_norm > self.tol and num_iter < self.max_iter:

            # calculate alpha.
            #alpha = 0.05
            alpha = self.exact_line_search(gfx,x)
            
            # update iterates.
            x_new = x + alpha * p
           
            gf_new = self.func_grad_(x_new)

            # calculate beta.
            if self.method == 'FR':
                beta = np.dot(gf_new, gf_new) / np.dot(gfx, gfx)
            else:
                raise ValueError('Method not implemented')
            
            # update everything.
            x = x_new
            gfx = gf_new
            p = -gfx + beta * p
            gfx_norm = np.linalg.norm(gfx)

            # collect the results.
            num_iter += 1
            curve_x.append(x)
            curve_fx.append(fx)
    
            if(self.verboose):
                print('Iteration {}: alpha = {:.4f}, residual = {}\n'.format(num_iter, alpha, gfx_norm))
            
        if num_iter == self.max_iter:
            print('CGD did not converge')

        return np.array(curve_x), np.array(curve_fx)

# Main function
def main():
    # Generate the matrix A.
    np.random.seed(0)
    A = np.random.randn(1000, 10)
    x0 = np.random.randn(10)
    funcs = Funcs(A)
    # Initialize the CGD algorithm.
    cgd = CGD(funcs.func_,funcs.func_grad_,funcs.exact_line_search,x0, 1e-4, 1000, method='FR', verboose=True)

    # Run the algorithm.
    x_ , fx_ = cgd.cgd()

    #plotting results.
    #plots = plot(funcs.func_)
    #X, Y, Z = plots.create_mesh()


main()







    