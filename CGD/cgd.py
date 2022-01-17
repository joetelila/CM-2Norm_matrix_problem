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
from scipy.optimize import line_search

class CGD:

    # Initialize the CGD algorithm
    def __init__(self,matrix_norm, func_, func_grad_,exact_line_search, x0, tol, max_iter, method='FR', verboose=False):
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
        self.matrix_norm = matrix_norm
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
        residuals = []
        errors = []
        error = abs(np.sqrt(abs(fx)) - self.matrix_norm) / abs(self.matrix_norm)
        errors.append(error)
        residuals.append(gfx_norm)    
        #if(self.verboose):
        #    print('Initial condition: fx = {:.4f}, x = {} \n'.format(fx, x))

        # Start the algorithm iterations.
        while gfx_norm > self.tol and num_iter < self.max_iter:

            # calculate alpha.
            #alpha = 0.05
            alpha = self.exact_line_search(gfx,x)
            if alpha < 1e-16:
                print('CGD did not converge')
                break
            # update iterates.
            x_new = x + alpha * p
            fx_new = self.func_(x_new)
            
        
            gf_new = self.func_grad_(x_new)
             
            # calculate error. 
            error = abs(np.sqrt(abs(fx_new)) - self.matrix_norm) / abs(self.matrix_norm)

            # calculate beta.
            if self.method == 'FR':
                beta = np.dot(gf_new, gf_new) / np.dot(gfx, gfx)
            elif self.method == 'PR':
                y_hat = gf_new - gfx
                beta = np.dot(gf_new, y_hat) / np.dot(gfx, gfx)
            elif self.method == 'HS':
                y_hat = gf_new - gfx
                beta = np.dot(y_hat, gf_new) / np.dot(y_hat, p)
            elif self.method == 'DY':
                y_hat = gf_new - gfx
                beta = np.dot(gf_new, gf_new) / np.dot(y_hat, p)
            else:
                raise ValueError('Method not implemented')
            
            # update everything.
            x = x_new
            fx = fx_new
            gfx = gf_new
            p = -gfx + beta * p
            gfx_norm = np.linalg.norm(gfx)

            # collect the results.
            num_iter += 1
            residuals.append(gfx_norm)
            errors.append(error)
            if(self.verboose):
                print('Iteration {}: error = {:.4f}, residual = {}\n'.format(num_iter, error, gfx_norm))
            
        if num_iter == self.max_iter:
            print('CGD did not converge')

        return residuals,errors, fx








    