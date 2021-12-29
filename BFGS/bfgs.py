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
# from scipy.optimize import line_search as off_shelf_line_search


class BFGS:

    # Initialize the BFGS algorithm
    def __init__(self, func_, func_grad_, line_search, x0, H0, tol, max_iter, method, line_search_args = dict(), verboose=False):
        '''
        Parameters
        ----------
            func_ : function, function to be minimized. 
                    f(x) = (x^T M x) / x^T x , where M = A^T A
        func_grad : function, Gradient of the function to be minimized. 
                    f'(x) = ((2 x (x^T M x)) / (x^T x)^2) - ((2 M x) / (x^T x))
       line_search: str, function used in the line search process
                    "A" for Armijo-type line search, 
                    "W" for Wolfe-type line search
               x0 : ndarray, initial guess. Can be set to be any numpy vector
               H0 : ndarray, initial (inverse) Hessian approximation. 
                    default as the Identity matrix
              tol : float, tolerance.
         max_iter : int , maximum number of iterations.
           method : str, optional, Method to be used to calculate B_k+1. 
                   "O" for Original BFGS,
                   "C" for  Cautious-BFGS  
 line_search_args : dict, optional, Arguments to be passed to the line_search function
          verbose : bool, optional, if True prints the progress of the algorithm.

        '''
        self.func_ = func_
        self.func_grad_ = func_grad_
        self.line_search = line_search
        self.x0 = x0
        self.H0 = H0
        self.tol = tol
        self.max_iter = max_iter
        self.method = method
        self.line_search_args = line_search_args
        self.verboose = verboose    

    # BFGS algorithm
    def bfgs(self):
        '''
        
        BFGS algorithm to solve the problem of estimating the matrix 2-norm as an
        unconstrained optimization.
        
        '''
        # initialize the variables.
        x = self.x0
        H = self.H0
        fx = self.func_(x)
        gfx = self.func_grad_(x)
        gfx_norm = la.norm(gfx)
        I = np.identity(x.shape[0])
        
        # constants for cautious update
        c_eps = 0.1

        # Collect the results.
        num_iter = 0
        residuals = []
        errors = []
        
        while gfx_norm > self.tol and num_iter < self.max_iter:

            # calculate p_k (direction for the line search)
            p = - np.dot(H, gfx)
            
            # calculate alpha (step length) FOCUS HERE
            line_search_results = self.line_search(self.func_, self.func_grad_, xk=x, pk=p, **self.line_search_args)
            
            if isinstance(line_search_results, tuple):
                alpha = line_search_results[0]
            else:
                alpha = line_search_results
            
            if alpha == None:
                alpha = 0
            
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
            # perform check for C-BFGS
            elif self.method == 'O' or (self.method == 'C' and ( np.dot(y.T, s) / np.linalg.norm(s)**2 ) > c_eps * ( np.linalg.norm(gfx)**c_alpha ) ):            
                # performs update of the H matrix
                ro = 1.0 / (np.dot(y, s))
                A1 = I - ro * s[:, np.newaxis] * y[np.newaxis, :]
                A2 = I - ro * y[:, np.newaxis] * s[np.newaxis, :]
                H_new = np.dot(A1, np.dot(H, A2)) + (ro * s[:, np.newaxis] * s[np.newaxis, :])
                
            else:
                H_new = H
            
            # update everything.
            x = x_new
            fx = fx_new
            gfx = gfx_new
            H = H_new # we could update H directly to save ops. 
            gfx_norm = la.norm(gfx)

            # collect the results.
            num_iter += 1
            residuals.append(gfx_norm)
            errors.append(error)
            if(self.verboose):
                print('Iteration {}: error = {:.4f}, residual = {}\n'.format(num_iter, error, gfx_norm))
        
        if(self.verboose):    
            if num_iter == self.max_iter:
                print('BFGS did not converge')
            else:
                print('BFGS did converge')

            print("Number of iterations: " + str(num_iter))
            print("The norm of the matrix is: " + str(np.sqrt(-fx)))
        
        return residuals,errors, fx








    
    