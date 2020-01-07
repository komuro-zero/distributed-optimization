import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 100
        self.m = 500
        self.r_i = 200
        self.iteration = 1000
        self.sparsity_percentage = 0.2
        self.lamb = 0.000001
        self.eta = 0.008
        self.B = 0.01
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.01
        self.w_noise = 10

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        extra_mc = self.pg_extra_mc_soft_nonconvex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00000007,0.0082,0.00000007*((1.1)**2),self.iteration,graph,w_all)
        plt.legend()
        plt.show()
        x = range(len(extra_mc))
        # plt.plot(x,extra,label = "extra")
        # plt.plot(x,extra_l1,label = "L1")
        plt.plot(x,extra_mc,label = "mc")
        # plt.plot(x,wdmc1,label = "distributed mc")
        plt.plot(x,w_star,color = "black")
        plt.legend()
        plt.show()
        # print(extra_mc)

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()