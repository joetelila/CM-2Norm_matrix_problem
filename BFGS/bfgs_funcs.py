import numpy as np

class Funcs:
    '''
  
        Class containing the functions used in the BFGS algorithm.
  
    '''
    
    def __init__(self, A):
        self.M = np.matmul(np.transpose(A), A)

    def func_(self,x):

        '''
            Function to be minimized. f(x) = (x^T M x) / x^T x , where M = A^T A
        '''
        #return np.matmul(np.matmul(np.transpose(x), self.M), x) / np.matmul(np.transpose(x), x)
        return (np.matmul(np.matmul(x.T, self.M), x) / np.matmul(x.T, x))*(-1)

    def func_grad_(self, x): # you can pass f(x) as a parameter. Later for optimization.
        '''
            Gradient of func_. f'(x) = ((2 x (x^T M x)) / (x^T x)^2) - ((2 M x) / (x^T x))
        '''
        return ((2 * x * np.matmul(x.T, np.matmul(self.M, x))) / (np.matmul(x.T, x))**2) - (2 * np.matmul(self.M, x)) / np.matmul(x.T, x)
  