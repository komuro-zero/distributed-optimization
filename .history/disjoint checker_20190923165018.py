a = [1]*40
b = [0]*60
a.extend(b)
c = [1]*40
d = [0]*60
d.extend(c)

print([a]*60)