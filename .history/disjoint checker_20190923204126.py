import numpy as np

m = 10
r_i = 1

a = [1]*r_i
print(a)
b = [0]*(m-r_i)
a.extend(b)
c = [1]*r_i
d = [0]*(m-r_i)
d.extend(c)
aa = [a]*int(m/2)
dd = [d]*int(m/2)
aa.extend(dd)
for i in range(len(aa)):
	aa[i][i] = 1
result = np.array(aa)
print(result)
