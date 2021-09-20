import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

eta = 0.2

def f(x, y):
    return x**2 - y**2

def l2(x, y):
    return x**2 + y**2

def l2_x(x, y):
    return (x**2)/16+np.sqrt(15)*x*y/8+(15*y**2)/16

def l2_y(x, y):
    return (15*x**2)/16-np.sqrt(15)*x*y/8+(y**2)/16

def f_2(x, y,a):
    return a*0.5*(x**2 + y**2)

def l1(x,y):
    return (abs(x) + abs(y)) 

def f_4(x, y):
    return 0.1*x**2

def y_l2(x, y,a):
    return 0.5*a*y**2

def infinity_infimal(x,y,eta):
    output = np.ones((101,101))
    for i in range(len(x)):
        for j in range(len(x[0])):
            x_now = x[i][j]
            y_now = y[i][j]
            if x_now>= 1/eta and y_now >= 1/eta:
                output[i][j] = 0.5*eta*(x_now-1/eta)**2 + 0.5*eta*(y_now-1/eta)**2
            if x_now <= -1/eta and y_now <= -1/eta:
                output[i][j] = 0.5*eta*(x_now+1/eta)**2 + 0.5*eta*(y_now+1/eta)**2
            if x_now >= 1/eta and y_now <= -1/eta:
                output[i][j] = 0.5*eta*(x_now-1/eta)**2 + 0.5*eta*(y_now+1/eta)**2
            if x_now <= -1/eta and y_now >= 1/eta:
                output[i][j] = 0.5*eta*(x_now+1/eta)**2 + 0.5*eta*(y_now-1/eta)**2
            elif x_now >= 1/eta and abs(y_now) <= 1/eta:
                output[i][j] = 0.5*eta*(x_now-1/eta)**2
            elif x_now <= -1/eta and abs(y_now) <= 1/eta:
                output[i][j] = 0.5*eta*(x_now+1/eta)**2
            elif abs(x_now) <= 1/eta and y_now >= 1/eta:
                output[i][j] = 0.5*eta*(y_now-1/eta)**2
            elif abs(x_now) <= 1/eta and y_now <= -1/eta:
                output[i][j] = 0.5*eta*(y_now+1/eta)**2
            elif abs(x_now) < 1/eta and abs(y_now) < 1/eta:
                output[i][j] =  0
    return output

def huber(x,y,a):
    output = np.ones((101,101))
    for i in range(len(x)):
        for j in range(len(x[0])):
            x_now = x[i][j]
            y_now = y[i][j]
            if abs(x[i][j]) + abs(y[i][j]) > a:
                output[i][j] = a*(abs(x[i][j]) + abs(y[i][j])-a/2)
            else:
                output[i][j] =  1/(2)*(abs(x[i][j]) + abs(y[i][j]))**2
    return output

def amc_huber(x,y,a):
    output = np.ones((101,101))
    for i in range(len(x)):
        for j in range(len(x[0])):
            x_now = x[i][j]
            y_now = y[i][j]
            if abs(x[i][j]) + abs(y[i][j]) > a:
                output[i][j] = a*(abs(x[i][j]) + abs(y[i][j])-a/2)
            else:
                output[i][j] =  1/(2)*(abs(x[i][j]))**2
    return output

def amc(x,y,a):
    output = np.ones((101,101))
    b =0.25*a**2
    print(b)
    for i in range(len(x)):
        for j in range(len(x[0])):
            x_now = x[i][j]
            distance = (x_now**2)**0.5
            if abs(distance) > 0.5*a:
                output[i][j] = b
            else:
                output[i][j] =  (a*abs(distance) - (distance**2))
    return output

X, Y = np.meshgrid(
        np.linspace(-10, 10, 101),
        np.linspace(-10, 10, 101),
    )
    
# Z = f(X, Y)
l2 = l2(X, Y)
l2_x = l2_x(X, Y)
l2_y = l2_y(X, Y)
l1_plot = l1(X,Y)
Z_4 = f_4(X,Y)
# y_l2 = y_l2(X,Y,eta)
# huber_1 = huber(X,Y,10)
# huber_2 = amc_huber(X,Y,5)
# mc_2 = amc(X,Y,15)
infinity_plot = infinity_infimal(X,Y,eta)
x = np.array([0,0])
u_1 = np.array([1/4,np.sqrt(15)/4])
u_2 = np.array([np.sqrt(15)/4,-1/4])

fig = plt.figure(figsize=(16, 6), facecolor="w")
ax_3d = fig.add_subplot(131, projection="3d")
# ax_3d.plot_surface(X, Y, l2)
ax_3d.set_xlim3d([-1.0, 1.0])

ax_3d.set_ylim3d([-1.0, 1.0])

ax_3d.set_zlim3d([0, 1])
ax_3d.quiver(x,x,x,u_1,u_2,x)

# fig = plt.figure(figsize=(16, 6), facecolor="w")
ax_3d = fig.add_subplot(132, projection="3d")
ax_3d.plot_surface(X, Y, l2_x)

ax_3d = fig.add_subplot(133, projection="3d")
ax_3d.plot_surface(X, Y, l2_y)

# fig = plt.figure(figsize=(16, 6), facecolor="w")
# ax_3d = fig.add_subplot(122, projection="3d")
# ax_3d.plot_surface(X, Y, mc_2+0.5*Z_4)

# ax = fig.add_subplot(122)
# contour = ax.contourf(X, Y, Z)
# fig.colorbar(contour)
plt.show()