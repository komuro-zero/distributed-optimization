import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
from matplotlib.backends.backend_pdf import PdfPages


np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):
        self.N = 10
        self.m = 100
        self.r_i = 20
        self.iteration =10000
        self.sparsity_percentage = 0.1
        self.lamb = 1.81/self.m
        self.eta = 0.00096
        self.B = 5.75/self.m
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 30

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        # self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        # self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        plt.rcParams["font.family"] = "Times New Roman" 
        pdf = PdfPages('extra_compare.pdf')


        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1,self.eta,self.rho,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.01,self.eta,self.rho,self.iteration,graph,w_all)
        # error,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,0.01*self.m,0.00657,self.m*0.0001,self.iteration)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.01*self.m,0.00657,self.m*0.0001,self.iteration,self.m)

        # extra = self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.012,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,selfx.m,self.r_i,3.83/self.m,0.012,self.rho,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,4.25/self.m,0.012,3.7/self.m,self.iteration,graph,w_all)

        
        # extra = self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,0.065,self.rho,self.iteration,graph,w_all)
        # extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.37/self.m,0.066,self.rho,self.iteration,graph,w_all)
        # prox_dgd = self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.053,self.rho,2,self.iteration,graph,w_all)
        # prox_dgd = self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.01,self.rho,2,self.iteration,graph,w_all)
        prox_dgd = self.prox_dgd(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,2/self.m,0.005,self.rho,2,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.7/self.m,0.067,2.25/self.m,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft_nonconvex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,1.807/self.m,0.00096,16.15/self.m,self.iteration,graph,w_all)
        # error,wcmc = self.centralized_mc_twin(U_all,d_all,w,w_star,L2,1.81,0.00096,5.75,self.iteration,self.m)
        # error_nonconvex,wcmcnc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,1.807,0.00096,16.15,self.iteration,self.m)
        
        # plt.hlines(error[-1],-10,self.iteration+10,color = "red",label = "Centralized Version with approximate MC penalty") 
        # plt.hlines(error_nonconvex[-1],-10,self.iteration+10,color = "purple", label = "Centralized Version with MC penalty") 
        plt.xlabel("Iterations")
        plt.ylabel("System Mismatch (dB)")
        # plt.plot(range(len(extra_mc)),extra_mc,label = 'PG-EXTRA with MC penalty')

        # extra_mc_non = self.pg_extra_mc_soft_nonconvex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.01,0.00657,0.0001,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.05,0.00657,0.0001,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.001,0.00657,0.0001,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0001,0.00657,0.0001,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00001,0.00657,0.0001,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.001,0.00657,0.00000001,self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.0001,0.00657,0.0000000000001,self.iteration,graph,w_all)

        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1,0.000657,0.1*(0.9**2),self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1,0.000657,0.1*(0.8**2),self.iteration,graph,w_all)
        # extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.1,0.00000657,self.rho,self.iteration,graph,w_all)
        plt.legend()
        plt.grid(which = "major")
        pdf.savefig()
        plt.show()
        pdf.close()
        x = range(len(extra_mc))
        plt.plot(x,wcmc,label = "cent")
        plt.plot(x,extra_mc,label = "pg extra")
        # plt.plot(x,extra_mc_non,label = "extra")
        # plt.plot(x,extra_l1,label = "L1")
        # plt.plot(x,extra_mc,label = "mc")
        # plt.plot(x,wdmc1,label = "distributed mc")
        plt.plot(x,w_star,color = "black")
        plt.legend()
        plt.show()
        # print(extra_mc)

if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()