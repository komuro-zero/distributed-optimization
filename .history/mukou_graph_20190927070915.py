import numpy as np

m = 10
r_i = 2
N = 3

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

def horizontal_checker(c,place,r_i):
    count = 0
    for row in c:
        if row[place] == 1:
            count += 1
    if count == r_i-1:
        result = False
    elif count < r_i-1:
        result = True
    else:
        print(c)
        print("faulty horizontal number")
        exit()
    return result




for i in range(m):
    checklist = make_checklist(r_i,m)
    connected = 0
    while connected < r_i:
        select = np.random.randint(len(checklist))
        connect_node = checklist[select]
        if c[i][connect_node] != 1 and horizontal_checker(c,connect_node,r_i):
            c[i][connect_node] = 1
            connected += 1
            c[connect_node][i] =1
            checklist.pop(select)
print(c)