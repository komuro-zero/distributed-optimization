import numpy as np

m = 10
r_i = 4

a = [1]*r_i
b = [0]*(m-r_i)
a.extend(b)
c = [1]*r_i
d = [0]*(m-r_i)
d.extend(c)
aa = [a]*int(m/2)
dd = [d]*int(m/2)
dd.extend(aa)
for i in range(len(dd)):
	dd[i][i] = 1
result = np.array(dd)
print(result)
print(result)