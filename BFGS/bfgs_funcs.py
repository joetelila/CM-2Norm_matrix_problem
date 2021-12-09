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
        return np.matmul(np.matmul(np.transpose(x), self.M), x) / np.matmul(np.transpose(x), x)

    def func_grad_(self, x): # you can pass f(x) as a parameter. Later for optimization.
        '''
            Gradient of func_. f'(x) = ((2 x (x^T M x)) / (x^T x)^2) - ((2 M x) / (x^T x))
        '''
        return ((2 * x * np.matmul(x.T, np.matmul(self.M, x))) / (np.matmul(x.T, x))**2) - (2 * np.matmul(self.M, x)) / np.matmul(x.T, x)
  

    def line_search(self, d,x):
        '''
            Find alpha that satisfies strong Wolfe conditions.

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



        a = (dTMx_ * dTd_)
        b = -((xTMx_ * dTd_) - (dTMd_ * xTx_))
        c = ((xTMx_ * xTd_) - (dTMx_ * xTx_))

        coef = np.array([a, b, c])
        roots = np.roots(coef).tolist()

        if(roots[0].real < 0 and roots[1].real < 0):
            return 0
        elif(roots[0].real < 0):
            return roots[1].real
        elif(roots[1].real < 0):
            return roots[0].real
        else:
            return roots[1].real if roots[1].real > roots[0].real else roots[0].real
