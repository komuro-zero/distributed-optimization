import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
import numpy.linalg as LA


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 60
        self.m = 100
        self.r_i = 80
        self.iteration = 1000
        self.sparsity_percentage = 0.2
        self.lamb = 2.3
        self.eta = 0.00657
        self.B = 3.73
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 20


    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.00655,self.iteration)
        error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,4,0.002849,self.iteration)
        error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,2,0.002849,self.iteration)
        error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.1,0.002849,self.iteration)
        error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,2.3,0.002849,1.6,self.iteration,self.m)
        plt.legend()
        plt.show()
        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        
        x = range(len(extra_l1))
        # plt.plot(x,extra,label = "extra")
        plt.plot(x,extra_l1,label = "L1")
        # plt.plot(x,extra_mc,label = "mc")
        # plt.plot(x,wdmc1,label = "distributed mc")
        plt.plot(x,w_star,color = "black")
        plt.legend()
        plt.show()
        print(extra_mc)

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()