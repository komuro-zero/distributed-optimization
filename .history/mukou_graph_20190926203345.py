import numpy as np

m = 100
r_i = 39
N = 30

c = np.eye(m) 
a = range(m)
for i in range(r_i-1):
    a[0:0] = range(m)
print(a.sort())