import numpy as np

m = 100
r_i = 2
N = 40

c = np.eye(m) 
for j in range(m):
	counter = 0
	while counter != r_i:
		random_num = np.random.randint(0,m-1)
		if c[j][random_num] != 1:
			c[j][random_num] = 1
			counter += 1

c_list = c.tolist()

def find_ones(row):
	result = [i for i, element in enumerate(row) if element == 1]

for row in c_list:
	print([i for i, element in enumerate(row) if element == 1])