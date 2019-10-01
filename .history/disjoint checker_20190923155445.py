import numpy as np
import random

m = 10
r_i = 4

c = np.eye(m) 
for j in range(m):
	counter = 0
	while counter != r_i:
		random_num = random.randint(0,m-1)
		if c[j][random_num] != 1:
			c[j][random_num] = 1
			counter += 1 
row_total = c[0] 
for i in range(row_total):
	if element != 0:
		for row in c:
			if row[i] != 0:
				row_total +=