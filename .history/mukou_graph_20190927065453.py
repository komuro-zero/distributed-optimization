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

#not done
def make_checklist(r_i,m):
    untidy_result = make_range(m)
    for i in range(r_i-1):
        untidy_result[0:0] = make_range(m)
    return sorted(untidy_result)

def horizontal_checker(c,place):
    count = 0
    for row in c:
        if row[place] == 1:
            count += 1
    return count



checklist = make_checklist(r_i,m)
connected = 0
for i in m:
    select = np.random.randint(r_i*m)
    connect_node = checklist[select]
    if i != connect_node and c[i][connect_node] != 1:
        c[i][connect_node] = 1
        connected += 1