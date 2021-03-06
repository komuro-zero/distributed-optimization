import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 100
        self.m = 1000
        self.r_i = 80
        self.iteration =20
        self.sparsity_percentage = 0.1
        self.lamb = 1.807
        self.eta = 0.00096
        self.B = 16
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 30

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,1.807,0.00096,16.15,self.iteration,self.m)

        plt.grid(which = "major")
        plt.xlabel("Iterations")
        plt.ylabel("System Mismatch (dB)")
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