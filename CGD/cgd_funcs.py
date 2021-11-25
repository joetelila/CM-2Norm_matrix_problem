import numpy as np

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
  
  def Griewank(self,xs):
    """Griewank Function"""
    d = len(xs)
    sqrts = np.array([np.sqrt(i + 1) for i in range(d)])
    cos_terms = np.cos(xs / sqrts)
    
    sigma = np.dot(xs, xs) / 4000
    pi = np.prod(cos_terms)
    return 1 + sigma - pi
  

  def exact_line_search(self, d,x):
        '''
        The value of α along the direction d is given as the root of the equation below:
        
        α^2 a + αb + c = 0 , where
        a = (d^T M x) / (d^Td)
        b = -((x^T M x) d^T d - (d^T M d) (x^T x))
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

        coef = np.array([a, -b, c])
        roots = np.roots(coef).tolist()

        if(roots[0].real < 0 and roots[1].real < 0):
            return 0
        elif(roots[0].real < 0):
            return roots[1].real
        elif(roots[1].real < 0):
            return roots[0].real
        else:
            return roots[1].real if roots[1].real > roots[0].real else roots[0].real
