import numpy as np

m = 100
r_i = 20
N = 40

c = np.eye(m) 
for j in range(m):
	counter = 0
	while counter != r_i:
		random_num = np.random.randint(0,m)
		if c[j][random_num] != 1:
			c[j][random_num] = 1
			counter += 1


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

def disjoint_checker(c)
	c_list = c.tolist()
	all_connection = all_connections(c_list)
	adjacent_matrix = delete_self(all_connection)
	check_node = adjacent_matrix[0]
	connected_node = adjacent_matrix[0]
	last_length = len(connected_node)
	check_node_box = []
	count = 1

	flag = True
	while flag:
		for i in check_node:
			check_node_box[0:0] = (all_connection[i])
		common_part = list(set(check_node_box) & set(connected_node))
		check_node = list(set(check_node_box) - set(common_part))
		connected_node[0:0] = check_node_box
		connected_node = list(set(connected_node))
		check_node_box = []
		now_length = len(connected_node)
		print(connected_node)
		if now_length == last_length:
			flag = False
		count += 1
		if count > 100 or len(connected_node) ==100:
			flag = False
		last_length = now_length
	return now_length