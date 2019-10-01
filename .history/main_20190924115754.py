import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_functions import functions


class distributed_updates(functions):


    def __init__(self):

        self.N = 30
        self.m = 100
        self.r_i = 40
        self.iteration = 100
        self.sparsity_percentage = 0.2
        self.lamb = 10**-2
        self.eta = 0.24
        self.rho = self.lamb*(20**2)
        if self.lamb/self.rho <= self.eta*self.lamb:
            print("faulty rho")
            exit()

    def make_variables(self):
        """
        w_star,U_all,d_allを返㝙
        """
        w = randn(self.N,1)
        w_star = self.w_star(self.N,self.sparsity_percentage)
        U_all = randn(self.m,self.N)
        d_all = np.dot(U_all,w_star)
        L2 = np.dot(w_star.T,w_star)

        return w,w_star,U_all,d_all,L2

    def run(self):
        w,w_star,U_all,d_all,L2 = self.make_variables()
        c = self.adjacency_matrix(self.m,self.r_i)
        w_all = self.make_w(self.m,self.N)
        
        #self.centralized_gradient_descent(U_all,d_all,w,w_star,L2,0.0035,self.iteration)
        #self.centralized_L1(U_all,d_all,w,w_star,L2,10**-4,0.0036,self.iteration)
        #self.centralized_mc(U_all,d_all,w,w_star,L2,5*10**-1,0.0036,5*10**-1*(23.57**2),self.iteration)
        self.distributed_gradient_descent(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,0.25,self.iteration,c,w_all)
        self.distributed_L1(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,10**-4,0.26,self.iteration,c,w_all)
        dmc = self.distributed_mc(U_all,d_all,w_star,L2,self.N,self.m,self.r_i,self.lamb,self.eta,self.rho,self.iteration,c,w_all)
        
        #error = [cgd[-1],cl1,cmc[-1],dgd[-1],dl1[-1],dmc[-1]] 
        return dmc[-1]#error

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
        error = simulation.run()
        cgd.append(error)
        print("iteration:",i)
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
    times = range(len(cgd))

    print("average:",sum(cgd)/len(cgd),"\n max:",max(cgd))
    #plt.bar(times,cgd)
    plt.legend()
    plt.show()