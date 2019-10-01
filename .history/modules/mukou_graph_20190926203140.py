import numpy as np

m = 100
r_i = 39
N = 30

c = np.eye(m) 
for j in range(m):
    counter = 0
    while counter != r_i:
        random_num = np.random.randint(0,m)
        if c[j][random_num] != 1:
            c[j][random_num] = 1
            counter += 1