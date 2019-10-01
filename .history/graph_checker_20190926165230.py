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
	return result

def all_connections(c_list):
	result = []
	for row in c_list:
		result.append(find_ones(row))
	return result

def delete_self(all_connection):
	for i in range(len(all_connection)):
		all_connection[i].remove(i)
	return all_connection

all_connection = all_connections(c_list)
print(all_connection)
adjacent_matrix = delete_self(all_connection)
print(adjacent_matrix)