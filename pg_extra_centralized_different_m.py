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
        self.lamb = 11.5
        self.eta = 0.000116
        self.B = 0.001
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.01
        self.w_noise = 10

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        # self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        
        #N100 m1000 sparsity30 wnoise30 2db
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.00092,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.55,0.00092,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.93,0.00092,6,self.iteration,self.m)

        #N100 m1000 sparsity10 wnoise30 3db
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.00092,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,1.05,0.00092,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,1.8,0.00092,6,self.iteration,self.m)

        #N100 m10000 sparsity10 wnoise30 2db 
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.000092,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,3.7,0.000092,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,4.45,0.000092,9.5,self.iteration,self.m)

        #N100 m1000 sparsity10 wnoise20 weaklysparse0.01 2dB
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.00092,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,3.2,0.00092,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,3.8,0.00092,11.14,self.iteration,self.m)

        #N100 m1000 sparsity10 wnoise30 weaklysparse0.01 0.5dB
        # error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.00092,self.iteration)
        # error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.7,0.00092,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.8,0.00092,2.5,self.iteration,self.m)

        #N100 m1000 sparsity10 wnoise10 weaklysparse0.01 2dB
        error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.00092,self.iteration)
        error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,11,0.00092,self.iteration)
        error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,11.5,0.00092,6,self.iteration,self.m)
        
        plt.legend()
        plt.show()
        x = range(len(wcmc))
        plt.plot(x,wcmc,label = "mc")
        plt.plot(x,wcl1,label = "l1")
        plt.plot(x,wcgd,label = "gd")
        plt.plot(x,w_star,color = "black")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()