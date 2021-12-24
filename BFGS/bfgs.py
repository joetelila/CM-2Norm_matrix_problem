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
from scipy.optimize import line_search as off_shelf_line_search


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
       line_search: function, function used in the line search process
                    armijo_line_search()
                    wolfe_line_search()
                    custom_line_search()
               x0 : ndarray, initial guess. Can be set to be any numpy vector
               B0 : ndarray, initial Hessian approximation. 
                    default as the Identity matrix
              tol : float, tolerance.
         max_iter : int , maximum number of iterations.
           method : str, optional, Method to be used to calculate B_k+1. 
                   "O" for Original BFGS,
                   "C" for  Cautious-BFGS  
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
        
        # constants for cautious update
        c_eps = 0.1
            

        # Collect the results.
        num_iter = 0
        residuals = []
        errors = []

        #if(self.verboose):
        #    print('Initial condition: fx = {:.4f}, x = {} \n'.format(fx, x))

        # Start the algorithm iterations.
        
        while gfx_norm > self.tol and num_iter < self.max_iter:

            # calculate p_k (direction for the line search)
            p = - np.matmul(la.inv(B), gfx)
            
            # calculate alpha (step length) FOCUS HERE
            #line_search_results = off_shelf_line_search(self.func_, self.func_grad_, x, p, c1=0.1, c2=0.49, maxiter=300)
            line_search_results = off_shelf_line_search(self.func_, self.func_grad_, x, p, maxiter=300)
            alpha = line_search_results[0]
            if alpha == None:
                alpha = 1
            
            # update iterates.
            x_new = x + alpha * p
            fx_new = self.func_(x_new)
            gfx_new = self.func_grad_(x_new)
            
            # compute s_k and y_k
            s = x_new - x
            y = gfx_new - gfx
             
            # calculate error.
            error = abs(fx_new - fx)

            # calculate c_alpha for cautious update
            if gfx_norm >= 1:
                 c_alpha = 0.01
            else:
                c_alpha = 3
            
            # calculate B_new.    
            if self.method not in ['O', 'C']:
                raise ValueError('Method not implemented')
            elif self.method == 'O' or (self.method == 'C' and ( np.dot(y, s) / np.linalg.norm(s)**2 ) > c_eps * ( np.linalg.norm(gfx)**c_alpha ) ):
                Bs = np.matmul(B, s)
                sB = np.matmul(s, B)
                BssB = np.dot(Bs, sB)
                sBs = np.dot(sB, s)
                yy = np.dot(y, y)
                ys = np.dot(y, s)
                
                B_new = B - BssB/sBs + yy/ys
            else:
                B_new = B
            
            # update everything.
            x = x_new
            fx = fx_new
            gfx = gfx_new
            B = B_new
            gfx_norm = np.linalg.norm(gfx)

            # collect the results.
            num_iter += 1
            residuals.append(gfx_norm)
            errors.append(error)
            if(self.verboose):
                print('Iteration {}: error = {:.4f}, residual = {}\n'.format(num_iter, error, gfx_norm))
            
        if num_iter == self.max_iter:
            print('BFGS did not converge')
        else:
            print('BFGS did converge')

        return residuals,errors, fx








    