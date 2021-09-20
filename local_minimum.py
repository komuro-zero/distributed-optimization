import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions



np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):

        self.N = 8
        self.m = 32
        self.r_i = 6
        self.iteration = 1000000
        self.sparsity_percentage = 1
        self.lamb = 0.31
        self.eta = 0.0045
        self.B = 0.1
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.00
        self.w_noise = 0

    def run(self):
        def local_minimum_function(x):
            return 3*x**4+4*x**3-12*x**2+x+16
        x = []
        y_1 = []
        y_2 = []
        y_3 = []
        y_4 = []
        y_5 = []
        y_all = []
        h = 100
        def quadratic_function(x,a,b,c):
            return a*x**2+b*x+c
        # matplotlib.rcParams['ps.useafm'] = True
        # matplotlib.rcParams['pdf.use14corefonts'] = True
        # matplotlib.rcParams['text.usetex'] = True 
        # pdf = PdfPages('local minimum.pdf')
        plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
        plt.rcParams['font.size'] = 10 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
        for i in range(h):
            x.append((i+1)/20-2.5)
            y_1.append(quadratic_function((i+1)/20-3,1,2,3))
            y_2.append(quadratic_function((i+1)/20-3,3,-4,1))
            y_3.append(quadratic_function((i+1)/20-3,6,2,5))
            y_4.append(quadratic_function((i+1)/20-3,1,-7,2))
            y_5.append(quadratic_function((i+1)/20-3,0.5,5,1))
            y_all.append(y_1[-1]+y_2[-1]+y_3[-1]+y_4[-1]+y_5[-1])
        plt.grid(which = "major")
        plt.plot(x,y_1)
        # plt.plot(x,y_2)
        # plt.plot(x,y_3)
        # plt.plot(x,y_4)
        # plt.plot(x,y_5)
        # plt.plot(x,y_all,color= "red")
        plt.show()

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()