import numpy as np
import matplotlib.pyplot as plt
from numpy.core.einsumfunc import _optimal_path
from numpy.random import *
from modules.distributed_regression import update_functions


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 100
        self.m = 200
        self.r_i = 20
        self.iteration = 1000
        self.sparsity_percentage = 0.2
        self.lamb = 1.1
        self.eta = 0.00093
        self.B = 2.4
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 30

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        # self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        # self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        # self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,1.1,0.00093,3.5,self.iteration,self.m)
        error_sin,w_sin = self.centralized_nonconvex_sin(U_all,d_all,w,w_star,L2,0.015,0.0001,0.007,self.iteration,self.m)
        error_mc,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,0.91,0.00093,19,self.iteration)
        # smallest_error = 0
        # for i in range(100):
        #     for j in range(100):
        #         lamb = 0.001 + 0.001*i
        #         rho = 0.001 + 0.001*j
        #         error, w = self.centralized_nonconvex_sin(U_all,d_all,w,w_star,L2,lamb,0.001,rho,self.iteration,self.m)
        #         if error[-1] < smallest_error:
        #             smallest_error = error[-1]
        #             optimal_lamb = lamb
        #             optimal_rho = rho
        #         print(i,j,optimal_lamb,optimal_rho)
        # print(optimal_lamb,optimal_rho)
        # smallest_error = 0
        # for i in range(100):
        #     for j in range(100):
        #         lamb = 0.01 + 0.01*i
        #         rho = 1 + 1*j
        #         error, w_mc = self.centralized_mc(U_all,d_all,w,w_star,L2,lamb,0.002,rho,self.iteration)
        #         if error[-1] < smallest_error:
        #             smallest_error = error[-1]
        #             optimal_lamb = lamb
        #             optimal_rho = rho
        #         print(i,j,optimal_lamb,optimal_rho)
        # print(optimal_lamb,optimal_rho)

        plt.grid(which = "major")
        plt.xlabel("Iterations")
        plt.ylabel("System Mismatch (dB)")
        plt.legend()
        plt.show()
        x  = range(len(w_star))
        # plt.plot(x,extra,label = "extra")
        # plt.plot(x,wcl1,label = "L1")
        plt.plot(x,wcmc,label = "mc")
        plt.plot(x,w_sin,label = "sin")
        plt.plot(x,w_star,color = "black")
        plt.legend()
        plt.show()
        

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()