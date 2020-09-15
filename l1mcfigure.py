import numpy as np
import matplotlib.pyplot as plt
import pywt

x = np.linspace(-0.75,0.75,500)
plt.plot(x,pywt.threshold(x,0.25,"soft"),label= "soft thresholding (L1)",color = "orange")
plt.plot(x,pywt.threshold_firm(x,0.2,0.25),label = "firm thresholding (MC)", color = "blue")
plt.plot(x,x,linestyle="dashed",color = "pink")
plt.grid(which = "major")
plt.legend()
plt.show()

def huber_function(X):
    result = np.zeros(X.shape)
    for i in range(len(X)):
        for j in range(len(X[i])):
            result[i][j] = min((0.5*(X[i][j])**2),(abs(X[i][j]-1)+0.5),(abs(X[i][j]+1)+0.5))
    return result
delta = 0.01
xrange = np.arange(-2, 2, delta)
yrange = np.arange(-2, 2, delta)
X, Y = np.meshgrid(xrange,yrange)


#軸の設定
plt.axis([-1.2, 1.2, -1.2, 1.2])
plt.gca().set_aspect('equal', adjustable='box')

#描画
Z=abs(X)+abs(Y)-1
W=abs(X)+ abs(Y) - huber_function(X)-huber_function(Y)-0.5
L1_penalty=plt.contour(X, Y, Z, [0],colors = "orange",label = "L1 norm")
mc_penalty=plt.contour(X, Y, W, [0],colors = "blue",label = "MC penalty")
plt.grid(which = "major")
plt.show()