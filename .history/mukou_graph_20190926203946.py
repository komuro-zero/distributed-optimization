import numpy as np

m = 100
r_i = 2
N = 30

c = np.eye(m) 

def make_range(m):
    result =[]
    for i in range(m):
        result.append(i)
    return result

def make_checklist(r_i,m):
    result = make_range(m)
    for i in range(r_i):
        result.append(make_range(m))
    result.sort()
    return result

make_checklist(r_i,m)