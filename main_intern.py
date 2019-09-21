import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from distributed_functions import functions


class distributed_updates(functions):

    def __init__(self):
        self.N = 30
        self.m = 100
        self.r_i = 50
        self.iteration = 1000
        self.sparsity_percentage = 0.7
        self.lamb = 10**-5
        self.eta = 0.0041
        self.rho = self.lamb*(0.00001**2)

    def make_variables(self):
        """
        w_star,U_all,d_allăčżă
        """
        get = functions()
        w = randn(self.N,1)
        w_star = get.w_star(self.N,self.sparsity_percentage)
        U_all = randn(self.m,self.N)
        d_all = np.dot(U_all,w_star)
        L2 = np.dot(w_star.T,w_star)

        return w,w_star,U_all,d_all,L2

    def run(self):
        w,w_star,U_all,d_all,L2 = self.make_variables()
        get = functions()
        get.centralized_gradient_descent(U_all,d_all,w,w_star,L2,self.eta,self.iteration)
        #get.centralized_L1(U_all,d_all,w,w_star,L2,self.lamb,self.eta,self.iteration)
        #get.centralized_mc(U_all,d_all,w,w_star,L2,self.lamb,self.eta,self.rho,self.iteration)
        get.distributed_gradient_descent(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.eta,self.iteration)
        #get.distributed_L1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,self.eta,self.iteration)
        #get.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,self.eta,self.rho,self.iteration)


if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()
    plt.legend()
    plt.show()