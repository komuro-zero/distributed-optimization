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
        self.r_i = 80
        self.iteration =50000   
        self.sparsity_percentage = 0.1
        self.lamb = 0.06
        self.eta = 0.0078
        self.B = 4
        self.rho = self.lamb*((self.B)**2)
        self.how_weakly_sparse = 0.0
        self.w_noise = 30

    def run(self):
        w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise)
        self.params_checker(self.rho,self.lamb,self.eta,U_all,self.B,self.m,self.N,graph)
        self.centralized_convexity_checker(self.B,self.lamb,U_all,self.N)
        
        #centralized
        # self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.0078,500)
        # self.centralized_L1(U_all,d_all,w,w_star,L2,0.0365,0.0078,500)
        # self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.04,0.0078,0.04,500,self.m)
        # self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,0.0637,0.0078,0.0405,500,self.m)
        plt.rcParams["font.family"] = "Times New Roman" 
        
        # main
        extra = self.extra(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,3.4*10**-3,self.rho,self.iteration,graph,w_all)
        extra_l1 = self.pg_extra_l1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.000365,3.4*10**-3,self.rho,self.iteration,graph,w_all)
        extra_mc = self.pg_extra_mc_soft(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.00045,3.4*10**-3,0.00044,self.iteration,graph,w_all)
        extra_mc = self.pg_extra_mc_soft_nonconvex(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.000637,3.4*10**-3,0.000405,self.iteration,graph,w_all)
        error,wcmc =self.centralized_mc_twin(U_all,d_all,w,w_star,L2,0.04,0.0078,0.04,1000,self.m)
        error_nonconvex,wcmcnc = self.centralized_mc_twin_nonconvex(U_all,d_all,w,w_star,L2,0.0637,0.0078,0.0405,1000,self.m)
        pdf = PdfPages('test.pdf')

        plt.hlines(error[-1],-10,self.iteration+10,color = "red",label = "Centralized Version with approximated MC penalty") 
        plt.hlines(error_nonconvex[-1],-10,self.iteration+10,color = "red",label = "Centralized Version with MC penalty") 
        # plt.xlabel("Iterations")
        # plt.ylabel("System Mismatch (dB)")
        # plt.legend()
        # plt.show()
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
        plt.xlabel("Iterations")
        plt.ylabel("System Mismatch (dB)")
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