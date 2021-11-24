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

class Funcs:
  '''
  
    Class containing the functions used in the CGD algorithm.
  
  '''
  def __init__(self, A):
    self.M = np.matmul(np.transpose(A), A)

  def func_(self,x):
      '''

        Function to be minimized.
        f(x) = (x^T M x) / x^T x , where M = A^T A
      
      '''
      return np.matmul(np.matmul(np.transpose(x), self.M), x) / np.matmul(np.transpose(x), x)

  def func_grad_(self, x): # you can pass f(x) as a parameter. Later for optimization.
      '''
        
        Gradient of func_.
        f'(x) = ((2 x (x^T M x)) / (x^T x)^2) - ((2 M x) / (x^T x))
      
      '''
      return ((2 * x * np.matmul(x.T, np.matmul(self.M, x))) / (np.matmul(x.T, x))**2) - (2 * np.matmul(self.M, x)) / np.matmul(x.T, x)
  
  def exact_line_search(self, x, d):
      '''
        The value of α along the direction d is given as the root of the equation below:
        
        α^2 a + αb + c = 0 , where
        a = (d^T M x) / (d^Td)
        b = (x^T M x) d^T d - (d^T M d) (x^T x)
        c = (x^T M x) x^T d - (d^T M x) (x^T x)
        
      '''
      
      dT_ = np.transpose(d)
      xT_ = np.transpose(x)

      dTd_ = np.matmul(dT_, d)
      xTx_ = np.matmul(xT_, x)
      xTd_ = np.matmul(xT_, d)

      Mx_ = np.matmul(self.M, x)
      Md_ = np.matmul(self.M, d)

      dTMx_ = np.matmul(dT_, Mx_)
      xTMx_ = np.matmul(xT_, Mx_)
      dTMd_ = np.matmul(dT_, Md_)

      a = dTMx_ * dTd_
      b = (xTMx_ * dTd_) - (dTMd_ * xTx_)
      c = (xTMx_ * xTd_) - (dTMx_ * xTx_)
      
      coeff = [a, -b, c]
      roots = np.roots(coeff).tolist()

      if(roots[0].real < 0 and roots[1].real < 0):
          return 0
      elif(roots[0].real < 0):
            return roots[1].real
      elif(roots[1].real < 0):
            return roots[0].real
      else:
          return roots[1].real if roots[1].real > roots[0].real else roots[0].real




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
        
        if(self.verboose):
            print('Initial condition: fx = {:.4f}, x = {} \n'.format(fx, x))

        # Start the algorithm iterations.
        while gfx_norm > self.tol and num_iter < self.max_iter:

            # calculate alpha.
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
            print(gfx_norm)
            if(self.verboose):
                print('Iteration {}: fx = {:.4f}, x = {} \n'.format(num_iter, fx, x))
        
        if num_iter == self.max_iter:
            print('CGD did not converge')
        else:
            print('\nSolution: \t fx = {:.4f}, x = {}'.format(fx, x))


# Main function
def main():
    # Generate the matrix A.
    A = np.random.randn(100, 100)
    x0 = np.random.randn(100)
    funcs = Funcs(A)
    # Initialize the CGD algorithm.
    cgd = CGD(funcs.func_,funcs.func_grad_,funcs.exact_line_search,x0, 1e-4, 1000, method='FR', verboose=False)

    # Run the algorithm.
    cgd.cgd()
main()







    