from numba import vectorize
import numpy as np

@vectorize(["float32(float32,float32)"],target="cuda")
def VectorToplama(a,b):
  return a+b
  
def main():
  N=320000000
  
  A=np.ones(N,dtype=np.float32)
  B=np.ones(N,dtype=np.float32)
  C=np.ones(N,dtype=np.float32)
  
  C=VectorToplama(A,B)
  
  print(C)
  
if __name__ == '__main__':
  main()
