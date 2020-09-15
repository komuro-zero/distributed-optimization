import numpy as np
import random 

m = 100
r_i = 19
r_i += 1
graph = np.eye(m)
list_num = list(range(m))
vertex_dict = dict(zip(list_num,[{i} for i in list_num]))
i = 0
while i < r_i:
    for edge_num in range(len(graph)):
        count = 0
        while len(vertex_dict[edge_num]) < i + 1:
            rand = random.randint(0,m-1)
            if rand != edge_num and rand not in vertex_dict[edge_num] and len(vertex_dict[rand]) < r_i:
                vertex_dict[edge_num].add(rand)
                vertex_dict[rand].add(edge_num)
            print(edge_num,rand,vertex_dict[edge_num])
            count += 1
            if count > 500:
                vertex_dict = dict(zip(list_num,[{i} for i in list_num]))
                i = -1
    i += 1
                


for i in vertex_dict:
    for j in vertex_dict[i]:
        graph[i][j] = 1


#DADMM 用
print(m*r_i/2)
graph_2 = np.zeros((int(m*(r_i-1)/2),m))
count = 0
for i in range(m):
    for j in vertex_dict[i]:
        if j > i:
            graph_2[count][i] = 1
            graph_2[count][j] = -1
            count += 1
print(vertex_dict[m-1],graph_2[-1])
