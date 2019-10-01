import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_functions_wj import functions


np.random.seed(0)
class distributed_updates(functions):


    def __init__(self):

        self.N = 40
        self.m = 100
        self.r_i = 30
        self.iteration = 100
        self.sparsity_percentage = 0.2
        #self.lamb = 10**-3
        #self.eta = 0.2
        #self.rho = self.lamb*(10**2)
        self.lamb = 0.1
        self.eta = 0.0028
        self.rho = self.lamb*(10**2)
        if self.lamb/self.rho <= self.eta*self.lamb:
            print("faulty rho")
            exit()

    def make_variables(self):
        """
        w_star,U_all,d_allを返㝙
        """
        how_weak = 0.1
        w = randn(self.N,1)
        w_star = self.w_star_weakly_sparse(self.N,self.sparsity_percentage,how_weak)
        U_all = randn(self.m,self.N)
        w_star_noise = w_star +randn(self.N,1)/10
        #w_star_noise = w_star_noise/np.dot(w_star_noise.T,w_star_noise)
        d_all = np.dot(U_all,w_star_noise)
        L2 = np.dot(w_star.T,w_star)

        return w,w_star,U_all,d_all,L2,w_star_noise

    def run(self):
        w,w_star,U_all,d_all,L2,w_star_noise = self.make_variables()
        c = self.adjacency_matrix(self.m,self.r_i)
        self.disjoint_checker(c,self.m)
        w_all = self.make_w(self.m,self.N)
        
        error_cgd,wcgd = self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.003,self.iteration)
        error,wcl1 = self.centralized_L1(U_all,d_all,w,w_star,L2,0.1,0.003,self.iteration)
        error,wcmc = self.centralized_mc(U_all,d_all,w,w_star,L2,self.lamb,self.eta,self.rho,self.iteration)
        #self.distributed_gradient_descent_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.25,self.iteration,c,w_all,wcgd)
        #self.distributed_L1_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,10**-4,0.24,self.iteration,c,w_all,wcl1)
        #self.distributed_mc_wj(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,self.eta,self.rho,self.iteration,c,w_all,wcmc)
        #self.distributed_gradient_descent(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.21,self.iteration,c,w_all)
        #self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,self.eta,self.rho,self.iteration,c,w_all)
        
        #error = [cgd[-1],cl1,cmc[-1],dgd[-1],dl1[-1],dmc[-1]] 
        #return error[-1]#w_star,w_star_noise,wcgd,wcl1,wcmc

"""
    #not completed
    def run_distributed(self):
        w,w_star,U_all,d_all,L2 = self.make_variables()  

        lamb_l1 = 9*10**-5
        lamb_mc = 10**-3
        eta_gd = 0.1
        eta_l1 = 0.09
        eta_mc = 0.01

        self.distributed(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,lamb_l1,lamb_mc,eta_gd,eta_l1,eta_mc,self.rho,self.iteration)
"""


if __name__ == "__main__":
    simulation = distributed_updates()
    cgd=cl1=cmc=dgd=dl1=dmc=[]
    for i in range(1):
        simulation.run()
        #cgd.append(error)
        """
        error = simulation.run()
        cgd.append(error[0])
        cl1.append(error[1])
        cmc.append(error[2])
        dgd.append(error[3])
        dl1.append(error[4])
        dmc.append(error[5])

        print("iteration:",i)
        """
    #times = range(len(cgd))

    #print("average:",sum(cgd)/len(cgd),"\n max:",max(cgd))
    #print("w_star:",w_star,"\nwcgd:",wcgd-w_star_noise,"\nwcl1:",wcl1-w_star_noise,"\nwcmc:",wcmc-w_star_noise)
    #plt.bar(times,cgd)
    plt.legend()
    plt.show()