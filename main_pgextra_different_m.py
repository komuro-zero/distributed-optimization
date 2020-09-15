import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression     import update_functions


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 100
        self.m = 1000
        self.r_i = 80
        self.iteration =20000
        self.sparsity_percentage = 0.1
        self.lamb = 1.69
        self.eta = 0.00002849
        self.B = 0.001
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 30

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.8/self.m,0.00092,6/self.m,self.iteration,graph,w_all)
        extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.05/self.m,0.00092,self.rho,self.iteration,graph,w_all)
        extra = self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.00092,self.rho,self.iteration,graph,w_all)


        plt.legend()
        plt.xlabel("iterations")
        plt.ylabel("Mean Square Error (dB)")
        plt.show()
        x = range(len(extra_l1))
        plt.plot(x,extra,label = "EXTRA")
        plt.plot(x,extra_l1,label = "PG-EXTRA L1")
        plt.plot(x,extra_mc,label = "PG-EXTRA MC")
        # plt.plot(x,wdmc1,label = "distributed mc")
        plt.plot(x,w_star,color = "black")
        plt.legend()
        plt.show()
        # print(extra_mc)

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()

    #l1 -44 extra -39 mc -44.7弱