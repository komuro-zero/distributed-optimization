import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from modules.distributed_regression import update_functions
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib



np.random.seed(0)
class distributed_updates(update_functions):


    def __init__(self):

        self.N = 10

        self.m = 10
        self.r_i = 3
        self.sparsity_percentage = 0.3
        self.how_weakly_sparse = 0.000
        self.w_noise = 30
        self.normal_distribution = True
        self.w_zero = True

    def run(self):

        # make settings
        setting_number = 100
        for i in range(setting_number):
            w,w_star,w_all,U_all,d_all,L2,graph = self.make_variables_noise_after_2(self.N,self.m,self.r_i,self.sparsity_percentage,self.how_weakly_sparse,self.w_noise,self.normal_distribution,self.w_zero)
            if i == 0:
                all_w_star = w_star
                all_U_all = U_all
                all_d_all = d_all
                all_graph = graph
            else:
                all_w_star = np.hstack((all_w_star,w_star))
                all_U_all = np.hstack((all_U_all,U_all))
                all_d_all = np.hstack((all_d_all,d_all))
                all_graph = np.hstack((all_graph,graph))

        np.savetxt(f"./samples/sample_wstar_{setting_number}_m={self.m}",all_w_star)
        np.savetxt(f"./samples/sample_U_all_{setting_number}_m={self.m}",all_U_all)
        np.savetxt(f"./samples/sample_d_all_{setting_number}_m={self.m}",all_d_all)
        np.savetxt(f"./samples/sample_graph_{setting_number}_m={self.m}",all_graph)
        # loaded_w_star =np.loadtxt(f"sample_wstar_{setting_number}_N={self.N}")
        # print(all_w_star)
        # print(loaded_w_star)
        # if all_w_star.all() == loaded_w_star.all():
        #     print("success")
        # print(all_w_star[:,1].reshape(len(all_w_star),1))

        # output settings
        # np.savetxt(f'parameters_N={self.N}_{setting_number}.txt',all_settings,delimiter=",")
        
        
if __name__ == "__main__":
    simulation = distributed_updates()
    simulation.run()