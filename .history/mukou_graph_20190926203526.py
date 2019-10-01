import numpy as np

m = 100
r_i = 2
N = 30

c = np.eye(m) 
a = range(m)
b = a
for i in range(r_i-1):
    a[0:0] = b 
print(a.sort())