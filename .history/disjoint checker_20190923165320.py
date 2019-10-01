import numpy as np

a = [1]*40
b = [0]*60
a.extend(b)
c = [1]*40
d = [0]*60
d.extend(c)
aa = [a]*50
dd = [d]*50
aa.extend(dd)
for i in range(len(aa)):
	aa[i][i] = 1
result = np.array(aa)
print(result)