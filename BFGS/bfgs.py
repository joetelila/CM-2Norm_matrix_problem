'''
  Authors: Dawit Anelay
           Marco Pitex
           Yohannis Telila
           
    Date : 09 / 12 / 2021 

  Implemetation of the BFGS algorithm to solve the problem of estimating
  the matrix 2-norm as an uncostrained optimization problem.

'''

# Imports
import numpy as np
from numpy import linalg as la
from scipy.optimize import line_search


class BFGS:

    # Initialize the BFGS algorithm
    def __init__(self, func_, func_grad_, line_search, x0, B0, tol, max_iter, method, verboose=False):
        '''
        Parameters
        ----------
            func_ : function, function to be minimized. 
                    f(x) = (x^T M x) / x^T x , where M = A^T A
        func_grad : function, Gradient of the function to be minimized. 
                    f'(x) = ((2 x (x^T M x)) / (x^T x)^2) - ((2 M x) / (x^T x))
               x0 : ndarray, initial guess. Can be set to be any numpy vector
               B0 : ndarray, initial Hessian approximation. 
              tol : float, tolerance.
         max_iter : int , maximum number of iterations.
           method : str, optional, Method to be used to calculate B_k+1. 
                   "O" for Original BFGS,
                   "C" for  Cautious-BFGS,  
          verbose : bool, optional, if True prints the progress of the algorithm.

        '''
        self.func_ = func_
        self.func_grad_ = func_grad_
        self.line_search = line_search
        self.x0 = x0
        self.B0 = B0
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.verboose = verboose
    

    # BFGS algorithm
    def bfgs(self):
        '''
        
        BFGS algorithm to solve the problem of estimating the matrix 2-norm as an
        unconstrained optimization.
        
        '''
        # initialize the variables.
        x = self.x0
        B = self.B0
        fx = self.func_(x)
        gfx = self.func_grad_(x)
        gfx_norm = np.linalg.norm(gfx)

        # Collect the results.
        num_iter = 0
        residuals = []
        errors = []

        #if(self.verboose):
        #    print('Initial condition: fx = {:.4f}, x = {} \n'.format(fx, x))

        # Start the algorithm iterations.
        while gfx_norm > self.tol and num_iter < self.max_iter:

            # calculate p_k
            p = np.dot(la.inv(B), gfx)
            print(p.shape)
            print(x.shape)
            
            # calculate alpha.
            #alpha = 0.05
            line_search_results = line_search(self.func_, self.func_grad_, x, p)
            #alpha = self.line_search(gfx,x)
            print(line_search_results)
            alpha = line_search_results[0]
            
            # update iterates.
            x_new = x + alpha * p
            fx_new = self.func_(x_new)
            gf_new = self.func_grad_(x_new)
            
            # compute s_k and y_k
            s = x_new - x
            y = gfx_new - gfx
             
            # calculate error.
            error = abs(fx_new - fx)

            # calculate beta.
            if self.method == 'O':
                Bs = np.matmul(B, s)
                Bss = np.matmul(Bs, s)
                BssB = np.matmul(Bss, B)
                sB = np.matmul(s, B)
                sBs = np.matmul(sB, s)
                yy = np.dot(y, y)
                ys = np.dot(y, s)
                
                B_new = B - BssB/sBs + yy/ys
            #elif self.method == 'C':
            #    B_new = 
            else:
                raise ValueError('Method not implemented')
            
            # update everything.
            x = x_new
            fx = fx_new
            gfx = gf_new
            B = B_new
            #p = -gfx + beta * p
            gfx_norm = np.linalg.norm(gfx)

            # collect the results.
            num_iter += 1
            residuals.append(gfx_norm)
            errors.append(error)
            if(self.verboose):
                print('Iteration {}: error = {:.4f}, residual = {}\n'.format(num_iter, error, gfx_norm))
            
        if num_iter == self.max_iter:
            print('BFGS did not converge')

        return residuals,errors, fx








    