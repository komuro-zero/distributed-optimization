import numpy as np
from numpy.lib.function_base import average
from numpy.linalg.linalg import norm
from numpy.random import *
import matplotlib.pyplot as plt
import copy
from modules.base_functions import base
import math
import numpy.linalg as LA
# import pywt


class update_functions(base):
	def centralized_gradient_descent(self,Ut,d,w,w_star,L2,eta,iteration):
		w = np.zeros_like(w)
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		U = Ut.T
		for i in range(iteration):
			w = w - eta*(np.dot(U,(np.dot(Ut,w)-d)))
			one_error = np.dot((w-w_star).T,w-w_star)[0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
		plt.plot(times,error,label = "centralized gradient descent")
		return error,w

	def centralized_nonconvex_sin(self,Ut,d,w,w_star,L2,lamb,eta,rho,iteration,m):
		w = np.zeros_like(w)
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		U = Ut.T
		# u_eig ,u_vec = np.linalg.eig(U@Ut)
		# eta = 2/((max(u_eig)).real+lamb*0.5*math.pi/rho)
		for i in range(iteration):
			w = w - eta*(np.dot(U,(np.dot(Ut,w)-d))+lamb*self.w_sin(w,rho))
			one_error = np.dot((w-w_star).T,w-w_star)[0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
		plt.plot(times,error,label = "centralized nonconvex sin")
		return error,w

	def centralized_L1(self,Ut,d,w_first,w_star,L2,lamb,eta,iteration):
		w = np.zeros_like(w_first)
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		U = Ut.T
		u_eig ,u_vec = np.linalg.eig(U@Ut)
		eta = (1/max(u_eig)).real
		for i in range(iteration):
			w = w - eta*(np.dot(U,(np.dot(Ut,w)-d)))
			for j in range(len(w)):
				if w[j] > 0 and eta*lamb < abs(w[j]):
					w[j] -= eta*lamb
				elif w[j] < 0 and eta*lamb < abs(w[j]):
					w[j] += eta*lamb
				else:
					w[j] = 0
			one_error = np.dot((w-w_star).T,w-w_star)[0][0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
		plt.plot(times,error,label = 'centralized_L1')
		return error,w
	
	def centralized_mc(self,Ut,d,w,w_star,L2,lamb,eta,rho,iteration):
		B = (rho/lamb)**0.5
		flag = True
		# flag = self.centralized_convexity_checker(B,lamb,Ut,len(w))
		w = np.zeros_like(w)
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)] 
		times = [0]
		one_error =0
		U = Ut.T
		u_eig ,u_vec = np.linalg.eig(U@Ut)
		eta = (1/max(u_eig)).real
		if flag:
			for i in range(iteration):
				w = w - eta*(np.dot(U,(np.dot(Ut,w)-d)))
				for j in range(len(w)):
					if abs(w[j]) < eta*lamb:
						w[j] = 0
					elif eta*lamb < abs(w[j]) and abs(w[j]) < lamb/rho:
						w[j] = w[j]*(abs(w[j])-eta*lamb)/(abs(w[j])*(1-eta*rho))
					elif lamb/rho <= abs(w[j]):
						w[j] = w[j]
					else:
						print("banana")
						print("eta*lamb",eta*lamb,"lamb/rho =",lamb/rho,"w = ",w[j])
				one_error = np.dot((w-w_star).T,w-w_star)[0]
				error.append(self.db(one_error,L2))
				times.append(i+1)
				one_error = 0
				# print(i)
			# plt.plot(times,error,label = "Centralized Optimization with MC penalty (Firm Thresholding)")
			return error,w
		else:
			return [0],None

	def centralized_partial_mc(self,Ut,d,w,w_star,L2,lamb,eta_input,rho,iteration,m):
		B = (rho/lamb)**0.5
		self.centralized_convexity_checker(B,lamb,Ut,len(w))
		w = np.zeros_like(w)
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		prox = np.zeros_like(w)
		Ut_norm = Ut/np.linalg.norm(Ut)
		U_norm = Ut_norm.T
		# diagonal = np.diag(U@Ut)
		proj = U_norm@Ut_norm
		U = Ut.T
		# proj = U@Ut
		# proj = np.eye(len(w_star))
		
		u_eig ,u_vec = np.linalg.eig(U@Ut)
		sorted_eig = sorted(list(u_eig))
		eta = (eta_input/max(u_eig)).real
		if m < len(w_star):
			rho_bound = sorted_eig[-m].real
		else:
			rho_bound = min(sorted_eig)
		for i in range(iteration):
			p_w = proj@w
			prox = copy.deepcopy(p_w)
			for j in range(len(prox)):
				if prox[j] > 0 and lamb/rho< abs(prox[j]):
					prox[j] -= lamb/rho
				elif prox[j] < 0 and lamb/rho < abs(prox[j]):
					prox[j] += lamb/rho
				else:
					prox[j] = 0
			w = w - eta*(U@(Ut@w-d)-rho*p_w+rho*prox)
			for j in range(len(w)):
				if w[j] > 0 and eta*lamb < abs(w[j]):
					w[j] -= eta*lamb
				elif w[j] < 0 and eta*lamb < abs(w[j]):
					w[j] += eta*lamb
				else:
					w[j] = 0
			one_error = np.dot((w-w_star).T,w-w_star)[0][0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
		plt.plot(times,error,label = 'Centralized Optimization with partial mc')
		return error,w

	def centralized_scad(self,Ut,d,w,w_star,L2,lamb,eta,rho,iteration):
		B = (rho/lamb)**0.5
		self.centralized_convexity_checker(B,lamb,Ut,len(w))
		w = np.zeros_like(w)
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)] 
		times = [0]
		one_error =0
		U = Ut.T
		for i in range(iteration):
			w = w - eta*(np.dot(U,(np.dot(Ut,w)-d)))
			for j in range(len(w)):
				if abs(w[j]) <= 2*lamb:
					w[j] = (w[j]/abs(w[j]))*max([abs(w[j])-lamb,0])
				elif 2*lamb < abs(w[j]) and abs(w[j]) <= lamb*rho:
					w[j] = ((rho-1)*w[j]-(w[j]/(abs(w[j])))*lamb*rho)/(rho-2)
				elif lamb*rho < abs(w[j]):
					w[j] = w[j]
				else:
					print("banana")
			one_error = np.dot((w-w_star).T,w-w_star)[0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
			one_error = 0
		plt.plot(times,error,label = "Centralized Optimization with scad penalty")
		return error,w

	def centralized_mc_twin_with_amc(self,Ut,d,w,w_star,L2,lamb,eta,rho,iteration,m):
		B = (rho/lamb)**0.5
		self.centralized_convexity_checker(B,lamb,Ut,len(w))
		w = np.zeros_like(w)
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		prox = np.zeros_like(w)
		U = Ut.T
		u_eig ,u_vec = np.linalg.eig(U@Ut)
		eta = (1/max(u_eig)).real
		for i in range(iteration):
			prox = copy.deepcopy(w)
			for j in range(len(prox)):
				if prox[j] > 0 and lamb/rho< abs(prox[j]):
					prox[j] -= lamb/rho
				elif prox[j] < 0 and lamb/rho < abs(prox[j]):
					prox[j] += lamb/rho
				else:
					prox[j] = 0
			w -= eta*(U@(Ut@w-d)-(rho/m)*U@Ut@w+rho*prox)
			for j in range(len(w)):
				if w[j] > 0 and eta*lamb < abs(w[j]):
					w[j] -= eta*lamb
				elif w[j] < 0 and eta*lamb < abs(w[j]):
					w[j] += eta*lamb
				else:
					w[j] = 0
			one_error = np.dot((w-w_star).T,w-w_star)[0][0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
		plt.plot(times,error,label = 'Centralized Optimization with MC penalty (Dual Soft)')
		return error,w
	
	def centralized_mc_twin_2(self,Ut,d,w,w_star,L2,lamb,eta,rho,iteration,m):
		B = (rho/lamb)**0.5
		self.centralized_convexity_checker(B,lamb,Ut,len(w))
		w = np.zeros_like(w)
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		prox = np.zeros_like(w)
		U = Ut.T
		for i in range(iteration):
			for j in range(len(w)):
				if w[j] > 0 and lamb*eta< abs(w[j]):
					prox[j] = w[j] - lamb*eta
				elif w[j] < 0 and lamb*eta < abs(w[j]):
					prox[j] = w[j] + lamb*eta
				else:
					prox[j] = 0
			w = w - eta*(U@(Ut@w-d)-rho*U@(Ut@w))-prox
			for j in range(len(w)):
				if w[j] > 0 and eta*lamb < abs(w[j]):
					w[j] -= eta*lamb
				elif w[j] < 0 and eta*lamb < abs(w[j]):
					w[j] += eta*lamb
				else:
					w[j] = 0
			one_error = np.dot((w-w_star).T,w-w_star)[0][0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
		plt.plot(times,error,label = 'Centralized version with approximate MC penalty')
		return error,w

	
	def centralized_mc_twin_b(self,Ut,d,w,w_star,L2,lamb,eta,rho,iteration,m):
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		prox = np.zeros_like(w)
		U = Ut.T
		rho, big_eig = self.U_eigenvalue(Ut)
		for i in range(iteration):
			delta_q = np.zeros_like(w).T
			for j in range(len(w)):
				if w[j] > 0 and lamb/rho< abs(w[j]):
					prox[j] = w[j] - lamb/rho
				elif w[j] < 0 and lamb/rho < abs(w[j]):
					prox[j] = w[j] + lamb/rho
				else:
					prox[j] = 0
			for ui in Ut:
				delta_q += ((ui@w)[0])*ui
			delta_q = ((1/m)*delta_q).T
			w = w - eta*(U@(Ut@w-d)-rho*delta_q+rho*prox)
			for j in range(len(w)):
				if w[j] > 0 and eta*lamb < abs(w[j]):
					w[j] -= eta*lamb
				elif w[j] < 0 and eta*lamb < abs(w[j]):
					w[j] += eta*lamb
				else:
					w[j] = 0
			one_error = np.dot((w-w_star).T,w-w_star)[0][0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
		plt.plot(times,error,label = 'centralized mc twin prox')
		return error,w


	def centralized_mc_twin_nonconvex(self,Ut,d,w,w_star,L2,lamb,eta,rho,iteration,m):
		B = (rho/lamb)**0.5
		self.centralized_convexity_checker(B,lamb,Ut,len(w))
		w = np.zeros_like(w)
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		prox = np.zeros_like(w)
		U = Ut.T
		for i in range(iteration):
			for j in range(len(w)):
				if w[j] > 0 and lamb/rho< abs(w[j]):
					prox[j] = w[j] - lamb/rho
				elif w[j] < 0 and lamb/rho < abs(w[j]):
					prox[j] = w[j] + lamb/rho
				else:
					prox[j] = 0
			
			w = w - eta*(U@(Ut@w-d)-rho*w+rho*prox)
			for j in range(len(w)):
				if w[j] > 0 and eta*lamb < abs(w[j]):
					w[j] -= eta*lamb
				elif w[j] < 0 and eta*lamb < abs(w[j]):
					w[j] += eta*lamb
				else:
					w[j] = 0
			one_error = np.dot((w-w_star).T,w-w_star)[0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
		# plt.plot(times,error,label = 'Centralized Optimization with MC penalty (Dual Soft)')
		return error,w


	def one_gradient_descent(self,Ut,d,w,eta,c_1):
		ui = np.reshape(Ut,(len(Ut),1))
		uti = ui.T
		wi = np.reshape(w,(len(w),1))
		wi = (ui@(uti@wi-d))
		print((ui@(uti@wi-d)).T[0],w)
		exit()
		return wi.T[0]

	def all_gradient_descent(self,Ut,w_next,d,w_all,eta,c_1):
		w_all_next = copy.deepcopy(w_all)
		for i in range(len(Ut)):
			w_all_next[i] = self.one_gradient_descent(Ut[i],d[i],w_all[i],eta,c_1)
		return w_all_next

	def one_L1(self,Ut,d,w,lamb,eta):
		U = Ut.T
		w = w - eta*((U*(np.dot(Ut,w)-d)))
		for j in range(len(w)):
			if w[j] > 0 and eta*lamb < abs(w[j]):
				w[j] -= eta*lamb
			elif w[j] < 0 and eta*lamb < abs(w[j]):
				w[j] += eta*lamb
			else:
				w[j] = 0
		return w
	
	def all_L1(self,Ut,w_next,d,w_all,lamb,eta):
		for i in range(len(Ut)):
			w_next[i] = self.one_L1(Ut[i],d[i],w_all[i],lamb,eta)
		return w_next

	def one_mc(self,Ut,d,w,lamb,eta,rho):
		U = Ut.T
		w = w - eta*((U*(np.dot(Ut,w)-d)))
		for j in range(len(w)):
			if abs(w[j]) <= eta*lamb:
				w[j] = 0
			elif eta*lamb < abs(w[j]) and abs(w[j]) < lamb/rho:
				w[j] = w[j]*(abs(w[j])-eta*lamb)/(abs(w[j])*(1-eta*rho))
			elif lamb/rho <= abs(w[j]):
				w[j] = w[j]
			else:
				print("banana")
		return w
	
	def all_mc(self,Ut,w_next,d,w_all,lamb,eta,rho):
		for i in range(len(Ut)):
			w_next[i] = self.one_mc(Ut[i],d[i],w_all[i],lamb,eta,rho)
		return w_next

	def distributed_gradient_descent(self,Ut,d,w_star,L2,N,m,r_i,eta,iteration,c,w_all):
		average_error = []
		w_all = np.zeros_like(w_all)
		w_all_iter = copy.deepcopy(w_all)
		c_1 = (1/(r_i+1))*c
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all,w_star,N,L2,m))
			w_all = c_1@w_all - eta*self.all_gradient_descent(Ut,w_all,d,w_all_iter,eta,c_1)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'new distributed gradient descent')
		return np.mean(w_all,axis = 0)

	def distributed_L1(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,iteration,c,w_all):
		average_error = []
		w_all_next = copy.deepcopy(w_all)
		w_all_iter = copy.deepcopy(w_all)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_iter,w_star,N,L2,m))
			w_all_next = self.all_L1(Ut,w_all_next,d,w_all_iter,lamb,eta)
			w_all_iter = (1/(r_i+1))*(c@w_all_next)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'new L1')
		#return average_error,w_all

	def distributed_mc(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		B = (rho/lamb)**0.5
		self.distributed_convexity_checker(B,lamb,Ut,len(w))
		average_error = []
		w_all_next = copy.deepcopy(w_all)
		w_all_iter = copy.deepcopy(w_all)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_iter,w_star,N,L2,m))
			w_all_next = self.all_mc(Ut,w_all_next,d,w_all_iter,lamb,eta,rho)
			w_all_iter = (1/(r_i+1))*(c@w_all_next)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'distributed mc')
		return np.mean(w_all_iter,axis = 0)

	def distributed_mc_compare(self,Ut,d,wcmc,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		size = np.dot(wcmc.T,wcmc)[0][0]
		print(size)
		w_all_next = copy.deepcopy(w_all)
		w_all_iter = copy.deepcopy(w_all)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_iter,wcmc,N,size,m))
			w_all_next = self.all_mc(Ut,w_all_next,d,w_all_iter,lamb,eta,rho)
			w_all_iter = (1/(r_i+1))*(c@w_all_next)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'mc compare centralized with decentralized')
		return np.mean(w_all_iter,axis = 0)

	def pg_extra_mc(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)

		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_share(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc(Ut,d,w_all_prox,lamb,eta,rho)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'PG-EXTRA with MC penatly')
		return np.mean(w_all_next,axis = 0)

	def pg_extra_mc_consensus_graph(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		consensus_error = []
		neighbor_consensus_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		I = np.eye(len(c))
		c_2 = (1/(r_i))*(c-I)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			w_all_next_average = np.reshape(np.mean(w_all_next,axis = 0),(len(w_star),1))
			L2_consensus = np.linalg.norm(w_all_next_average, ord=2)**2
			if i == 0:
				average_error.append(0)
				consensus_error.append(0)
				neighbor_consensus_error.append(0)
			elif i != 0:
				average_error.append(self.error_distributed(w_all_next,w_star,N,L2,m))
				consensus_error.append(self.error_distributed(w_all_next,w_all_next_average,N,L2_consensus,m))
				neighbor_consensus_error.append(self.error_consensus(w_all_next,N,L2_consensus,m,c_2))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_share(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc(Ut,d,w_all_prox,lamb,eta,rho)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		# plt.plot(times,neighbor_consensus_error,label = 'without consensus term')
		plt.plot(times,consensus_error,label = 'PG-EXTRA with MC penalty')
		# plt.plot(times,average_error,label = 'PG-EXTRA with MC penalty')
		return np.mean(w_all_next,axis = 0)
	
	def extra_setup(self,c,r_i,Ut,mu,eta_multitude,rho,decay):
		I = np.eye(len(c))
		c_2 = (1/(r_i))*(c-I)
		graph_eig,graph_vec = LA.eig(I - c_2)
		graph_eig_list = sorted(list(graph_eig))
		mu = rho/((graph_eig[1]).real)
		pre_c = decay*(1/((r_i+1)))*(c-np.eye(len(c)))
		c_1 = pre_c + np.eye(len(c))*(1-r_i*decay/((r_i+1)))
		c_tilde = (1/2)*(np.eye(len(c_1))+c_1)
		kappa = max(list((Ut.T@Ut).diagonal()))+mu
		c_til_val, c_til_vec = np.linalg.eig(c_tilde)
		eta = eta_multitude*2*min(c_til_val)/kappa
		# print("============================================================\n","eta = ",eta,"\n============================================================\n")
		v_val, v_vec = np.linalg.eig((I -c_2).T@(I -c_2))
		u_val, u_vec = np.linalg.eig(Ut.T@Ut)
		v_2_min = sorted(list(v_val))[1]
		c_val, c_vec = np.linalg.eig((c_2).T@(c_2))
		v_2_min = mu*v_2_min
		tau_eta_sup = min(v_2_min,min(u_val))
		# if rho > tau_eta_sup:
			# print("rho should be smaller than ",tau_eta_sup)
			# exit()
		# print("rho can be as big as ",tau_eta_sup)
		return I,c_1,c_2,c_tilde,eta
	
	def extra_setup_for_lambda_2_search(self,c,r_i,mu):
		I = np.eye(len(c))
		c_2 = (1/(r_i))*(c-I)
		v_val, v_vec = np.linalg.eig((I -c_2).T@(I -c_2))
		v_2_min = sorted(list(v_val))[1]
		return v_2_min*mu

	def extra_setup_2(self,c,r_i,Ut,mu,eta_multitude,rho,decay):
		I = np.eye(len(c))
		c_2 = (1/(r_i))*(c-I)
		pre_c = decay*(1/((r_i+1)))*(c-np.eye(len(c)))
		c_1 = pre_c + np.eye(len(c))*(1-r_i*decay/((r_i+1)))
		c_tilde = (1/2)*(np.eye(len(c_1))+c_1)
		kappa = max(list((Ut.T@Ut).diagonal()))+mu
		c_til_val, c_til_vec = np.linalg.eig(c_tilde)
		eta = eta_multitude*2*min(c_til_val)/kappa
		print("============================================================\n","eta = ",eta,"\n============================================================\n")
		v_val, v_vec = np.linalg.eig(mu*(I -c_2).T@(I -c_2))
		u_val, u_vec = np.linalg.eig(Ut.T@Ut)
		v_2_min = sorted(list(v_val))[1]
		tau_eta_sup = min(v_2_min,min(u_val))
		if rho > tau_eta_sup:
			print("rho should be smaller than ",tau_eta_sup)
			exit()
		print("rho can be as big as ",tau_eta_sup)
		return I,c_1,c_2,c_tilde,eta ,min(c_til_val)

	def pg_extra_mc_consensus_violation(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta_multitude,rho,iteration,c,w_all,mu,decay):
		average_error = []
		consensus_error = []
		neighbor_consensus_error = []

		w_all = np.zeros_like(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		I,c_1,c_2,c_tilde,eta = self.extra_setup(c,r_i,Ut,mu,eta_multitude,rho,decay)
		print(eta)
		for i in range(iteration):
			w_all_next_average = np.reshape(np.mean(w_all_next,axis = 0),(len(w_star),1))
			L2_consensus = np.linalg.norm(w_all_next_average, ord=2)
			consensus_error.append(self.error_distributed(w_all_next,w_all_next_average,N,L2_consensus,m))
			neighbor_consensus_error.append(self.error_consensus(w_all_next,N,L2_consensus,m,c_2))
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_consensus_violation_share(Ut,d,c_1,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i,I,rho,m,c_2,mu)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc(Ut,d,w_all_prox,lamb,eta,rho)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		# plt.plot(times,average_error,label = 'DPD with MC-CPP ('+r"$\kappa = $"+ f"{mu}"+r"$, \zeta = $"+f"{decay})")
		if mu == 0:
			plt.plot(times,average_error,label = 'w/o CPP ('+r"$r = $"+ f"{r_i}, "+r"$\zeta = $"+f"{decay})")
		else:
			plt.plot(times,average_error,label = 'w/ CPP ('+r"$r = $"+ f"{r_i}, "+r"$\zeta = $"+f"{decay})")
		# plt.plot(times,average_error,label = 'DPD with MC-CPP ('+r"$r = $"+ f"{r_i}, "+r"$\kappa = $"+ f"{mu}"+r"$, \zeta = $"+f"{decay})")
		# print(w_star,w_all_next[0])
		# plt.plot(times,consensus_error,label = 'Consensus: D-TwiX with '+r"$\kappa = $"+ f"{mu}"+r"$, \zeta = $"+f"{decay}")
		# if mu ==0:
		# 	plt.plot(times,consensus_error,label = 'w/o CPP ('+r"$r = $"+ f"{r_i}, "+r"$, \zeta = $"+f"{decay})")
		# else:
		# 	plt.plot(times,consensus_error,label = 'w/ CPP ('+r"$r = $"+ f"{r_i}, "+r"$, \zeta = $"+f"{decay})")
		# plt.plot(times,consensus_error,label = 'Consensus: DPD with MC-CPP ('+r"$r = $"+ f"{r_i}, "+r"$\kappa = $"+ f"{mu}"+r"$, \zeta = $"+f"{decay})")
		# plt.plot(times,neighbor_consensus_error,label = 'Neighbor Consensus: D-TwiG with '+r"$\mu = $"+ f"{mu}"+f", m_param = "+f"{decay}")
		return np.mean(w_all_next,axis = 0)
	
	def pg_extra_mc_consensus_violation_animation(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta_multitude,rho,iteration,c,w_all,mu,decay):
		average_error = []
		w_all_next_list =[]
		consensus_error = []
		neighbor_consensus_error = []

		w_all = np.zeros_like(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		I,c_1,c_2,c_tilde,eta = self.extra_setup(c,r_i,Ut,mu,eta_multitude,rho,decay)
		print(eta)
		for i in range(iteration):
			w_all_next_average = np.reshape(np.mean(w_all_next,axis = 0),(len(w_star),1))
			L2_consensus = np.linalg.norm(w_all_next_average, ord=2)
			consensus_error.append(self.error_distributed(w_all_next,w_all_next_average,N,L2_consensus,m))
			neighbor_consensus_error.append(self.error_consensus(w_all_next,N,L2_consensus,m,c_2))
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_consensus_violation_share(Ut,d,c_1,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i,I,rho,m,c_2,mu)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc(Ut,d,w_all_prox,lamb,eta,rho)
			w_all_next_list.append(w_all_next[:,0:3])
		# times = range(len(average_error))
		# plt.plot(times,average_error,label = 'CPP param ('+r"$\mu = $"+ f"{mu})")
		return np.mean(w_all_next,axis = 0),w_all_next_list

	def pg_extra_mc_consensus_violation_fixed_tau_eta(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta_multitude,rho,iteration,c,w_all,mu,decay,eigenvalue):
		average_error = []
		consensus_error = []
		neighbor_consensus_error = []

		w_all = np.zeros_like(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		I,c_1,c_2,c_tilde,eta = self.extra_setup(c,r_i,Ut,mu,eta_multitude,rho,decay)
		for i in range(iteration):
			w_all_next_average = np.reshape(np.mean(w_all_next,axis = 0),(len(w_star),1))
			L2_consensus = np.linalg.norm(w_all_next_average, ord=2)
			consensus_error.append(self.error_distributed(w_all_next,w_all_next_average,N,L2_consensus,m))
			neighbor_consensus_error.append(self.error_consensus(w_all_next,N,L2_consensus,m,c_2))
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_consensus_violation_share(Ut,d,c_1,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i,I,rho,m,c_2,mu)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc(Ut,d,w_all_prox,lamb,eta,rho)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		# plt.plot(times,average_error,label = 'D-TwiG with '+r"$\mu = $"+ f"{mu}"+r"$, \zeta = $"+f"{decay}")
		if mu == 0:
			plt.plot(times,average_error,label = 'w/o CPP '+r"$r = $"+ f"{r_i}, "+r"$\lambda_2( \textit{\textbf{L}} ) = $"+f"{round(eigenvalue,2)}")
		else:
			plt.plot(times,average_error,label = 'w/ CPP '+r"$r = $"+ f"{r_i}, "+r"$\lambda_2( \textit{\textbf{L}} ) = $"+f"{round(eigenvalue,2)}")
		# plt.plot(times,average_error,label = 'DPD with '+r"$r = $"+ f"{r_i}, "+r"$\kappa = $"+ f"{mu}"+r"$, \lambda_2( \textit{\textbf{L}} ) = $"+f"{round(eigenvalue,2)}")
		# print(w_star,w_all_next[0])
		# plt.plot(times,consensus_error,label = 'Consensus: D-TwiG with '+r"$\mu = $"+ f"{mu}"+r"$, \zeta = $"+f"{decay}")
		# plt.plot(times,consensus_error,label = 'Consensus: D-TwiG with '+r"$r_i = $"+ f"{r_i}, "+r"$\mu = $"+ f"{mu}"+r"$, \zeta = $"+f"{decay}")
		# plt.plot(times,neighbor_consensus_error,label = 'Neighbor Consensus: D-TwiG with '+r"$\mu = $"+ f"{mu}"+f", m_param = "+f"{decay}")
		return np.mean(w_all_next,axis = 0)

	def pg_extra_scad_consensus_violation(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		I = np.eye(len(c))
		c_1 = (1/(r_i+1))*c
		c_2 = (1/(r_i))*(c-I)
		c_tilde = (1/2)*(np.eye(len(c_1))+c_1)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_consensus_violation_share(Ut,d,c_1,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i,I,rho,m,c_2,1)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_scad(Ut,d,w_all_prox,lamb,eta,rho)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'PG-EXTRA with consensus promoting term and SCAD')
		return np.mean(w_all_next,axis = 0)

	def pg_extra_scad(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		I = np.eye(len(c))
		c_1 = (1/(r_i+1))*c
		c_2 = (1/(r_i))*(c-I)
		c_tilde = (1/2)*(np.eye(len(c_1))+c_1)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_share(Ut,d,c_1,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_scad(Ut,d,w_all_prox,lamb,eta,rho)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'PG-EXTRA with SCAD')
		return np.mean(w_all_next,axis = 0)

	def pg_extra_mc_consensus_violation_consensus_graph(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,mu):
		consensus_error = []
		neighbor_consensus_error = []
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		I = np.eye(len(c))
		c_1 = (1/(r_i+1))*c
		c_2 = (1/(r_i))*(c-I)
		c_tilde = (1/2)*(np.eye(len(c_1))+c_1)
		for i in range(iteration):
			w_all_next_average = np.reshape(np.mean(w_all_next,axis = 0),(len(w_star),1))
			L2_consensus = np.linalg.norm(w_all_next_average, ord=2)
			if i == 0:
				average_error.append(0)
				consensus_error.append(0)
				neighbor_consensus_error.append(0)
			elif i != 0:
				consensus_error.append(self.error_distributed(w_all_next,w_all_next_average,N,L2_consensus,m))
				average_error.append(self.error_distributed(w_all_next,w_star,N,L2,m))
				neighbor_consensus_error.append(self.error_consensus(w_all_next,N,L2_consensus,m,c_2))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_consensus_violation_share(Ut,d,c_1,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i,I,rho,m,c_2,mu)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc(Ut,d,w_all_prox,lamb,eta,rho)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		plt.plot(times,neighbor_consensus_error,label = 'with consensus term')
		# plt.plot(times,consensus_error,label = r'D-TwiG with ' + r"$\mu$ = " + rf"{mu}")
		# plt.plot(times,average_error,label = 'PG-EXTRA with consensus violation MC penalty')
		return np.mean(w_all_next,axis = 0)

	def pg_extra_mc_consensus_violation_neighbor_consensus_graph(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,mu):
		consensus_error = []
		neighbor_consensus_error = []
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		I = np.eye(len(c))
		c_1 = (1/(r_i+1))*c
		c_2 = (1/(r_i))*(c-I)
		c_tilde = (1/2)*(np.eye(len(c_1))+c_1)
		for i in range(iteration):
			w_all_next_average = np.reshape(np.mean(w_all_next,axis = 0),(len(w_star),1))
			L2_consensus = np.linalg.norm(w_all_next_average, ord=2)
			if i == 0:
				average_error.append(0)
				consensus_error.append(0)
				neighbor_consensus_error.append(0)
			elif i != 0:
				consensus_error.append(self.db(np.mean(np.linalg.norm(w_all_next-(I-c_2)@w_all_next,ord=2,axis = 1)**2),L2*m))
				average_error.append(self.error_distributed(w_all_next,w_star,N,L2,m))
				neighbor_consensus_error.append(self.error_consensus(w_all_next,N,L2_consensus,m,c_2))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_consensus_violation_share(Ut,d,c_1,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i,I,rho,m,c_2,mu)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc(Ut,d,w_all_prox,lamb,eta,rho)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		# plt.plot(times,neighbor_consensus_error,label = 'with consensus term')
		plt.plot(times,consensus_error,label = 'neighbor consensus')
		# plt.plot(times,average_error,label = 'PG-EXTRA with consensus violation MC penalty')
		return np.mean(w_all_next,axis = 0)
	
	def pg_extra_mc_convex(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)

		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_share(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc_convex(Ut,d,w_all_prox,lamb,eta,rho)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'PG-EXTRA with MC penalty convex')
		return np.mean(w_all_next,axis = 0)

	def atc_pg_extra_mc(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.atc_extra_l1_share(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc(Ut,d,w_all_prox,lamb,eta,rho)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'ATC-PG-EXTRA with MC penalty')
		return np.mean(w_all_next,axis = 0)
	
	def NEXT_with_MC_penalty(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,tau,inner_iteration):
		average_error = []
		w_all = np.zeros_like(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_tilde = copy.deepcopy(w_all)
		w_all_tilde_prox = copy.deepcopy(w_all)
		z = copy.deepcopy(w_all)
		y = self.gradient_soft(Ut,w_all,d,lamb,eta,rho,m)
		pi = m*y - y
		w_all_before = copy.deepcopy(w_all)
		I = np.eye(len(c))
		c_1 = (1/(r_i+1))*c
		c_2 = (1/(r_i))*(c-I)
		c_tilde = (1/2)*(np.eye(len(c_1))+c_1)
		outer_eta = 0.01
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all,w_star,N,L2,m))
			for j in range(inner_iteration):
				w_all_tilde_prox = w_all_tilde - eta*((pi + self.gradient_NEXT(Ut,w_all,w_all_tilde,d,lamb,eta,rho,m,tau)))
				w_all_tilde = self.all_extra_L1(Ut,d,w_all_tilde_prox,lamb,eta)
				# w_all_tilde = self.l1_projection(Ut,d,w_all_tilde_prox,L2*1.95,1)
			z = w_all + outer_eta*(w_all_tilde - w_all)
			w_all_before = copy.deepcopy(w_all)
			w_all = c_1@z
			y = c_1@y + (self.gradient_soft(Ut,w_all,d,lamb,eta,rho,m) - self.gradient_soft(Ut,w_all_before,d,lamb,eta,rho,m))
			pi = m*y - self.gradient_soft(Ut,w_all,d,lamb,eta,rho,m)
			if i %100 == 0:
				print(f"iteration: {i}")
			outer_eta = outer_eta*(1-0.01*outer_eta)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'NEXT with MC penalty')
		return np.mean(w_all,axis = 0),average_error
	
	def NEXT_with_MC_penalty_2(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,inner_eta,tau,beta):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_tilde = copy.deepcopy(w_all)
		w_all_tilde_prox = copy.deepcopy(w_all)
		z = copy.deepcopy(w_all)
		y = self.gradient_soft(Ut,w_all,d,lamb,eta,rho,m)
		pi = m*y - y
		w_all_before = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all,w_star,N,L2,m))
			for j in range(100):
				w_all_tilde_prox = w_all_tilde - inner_eta*(pi + self.gradient_NEXT(Ut,w_all,w_all_tilde,d,lamb,eta,rho,m,tau))
				w_all_tilde_prox = self.all_extra_L1(Ut,d,w_all_tilde_prox,eta,lamb)
				w_all_tilde = self.l1_projection(Ut,d,w_all_tilde_prox,L2*1.95,1)
			z = w_all + (eta/((i + 1)**beta))*(w_all_tilde - w_all)
			eta = eta*(1-0.5*eta)
			w_all_before = copy.deepcopy(w_all)
			w_all = c@z
			y = c@y + (self.gradient_soft(Ut,w_all,d,lamb,eta,rho,m) - self.gradient_soft(Ut,w_all_before,d,lamb,eta,rho,m))
			pi = m*y - self.gradient_soft(Ut,w_all,d,lamb,eta,rho,m)
			print(f"iteration: {i}")
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'NEXT with MC penalty 2')
		return np.mean(w_all,axis = 0)


	def pg_extra_approximate_mc(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,decay,graph_plot=True):
		average_error = []
		# average_variance = []
		# average_convergence = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		w_all_prox_before = copy.deepcopy(w_all_prox)
		
		pre_c = decay*(1/((r_i+1)))*(c-np.eye(len(c)))
		c = pre_c + np.eye(len(c))*(1-r_i*decay/((r_i+1)))
		c_tilde = (1/2)*(np.eye(len(c))+c)
		c_tilde_min = min(LA.eig(c_tilde)[0])
		lip = max(LA.eig(Ut.T@Ut)[0])
		Ls = (1-rho)*lip+ rho
		eta = 2*c_tilde_min/Ls
		rho = 1/lamb
		# print(eta)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			# average_convergence.append(LA.norm(w_all_before-w_all_next))
			# average_variance.append(np.var(w_all_before,axis = 1))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			if i == 0:
				w_all_prox = c@w_all_before-eta*(self.gradient_soft(Ut,w_all_before,d,lamb,eta,rho,m))
			else:
				w_all_prox = c@w_all_next + w_all_prox_before - c_tilde@w_all_before - eta*(self.gradient_soft(Ut,w_all_next,d,lamb,eta,rho,m)-self.gradient_soft(Ut,w_all_before,d,lamb,eta,rho,m))
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,eta,lamb)
			# if i % 100 == 0:
			# 	print(f"iteration: {i}")
		times = range(len(average_error))
		if graph_plot:
			plt.plot(times,average_error,label = "PG-EXTRA with AMC penalty")
		# plt.title("convergence over iteration")
		# plt.show()
		return average_error,np.mean(w_all_before,axis = 0)
	
	def pg_extra_partial_mc(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta_input,rho,iteration,c,w_all,decay,graph=True,animation=False):
		average_error = []
		# average_variance = []
		# average_convergence = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		w_all_prox_before = copy.deepcopy(w_all_prox)
		
		pre_c = decay*(1/((r_i+1)))*(c-np.eye(len(c)))
		c = pre_c + np.eye(len(c))*(1-r_i*decay/((r_i+1)))
		c_tilde = (1/2)*(np.eye(len(c))+c)
		c_tilde_min = min(LA.eig(c_tilde)[0])
		largest_rho = 1 - lamb
		if self.partial_convexity_checker(largest_rho,Ut) and largest_rho > 0:
			eta = self.partial_eta_calculator(Ut,eta_input,c_tilde_min,lamb,largest_rho)
			w_all_next_list = []
			for i in range(iteration):
				average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
				# average_convergence.append(LA.norm(w_all_before-w_all_next))
				# average_variance.append(np.var(w_all_before,axis = 1))
				w_all_prox_before = copy.deepcopy(w_all_prox)
				if i == 0:
					w_all_prox = c@w_all_before-eta*(self.gradient_partial(Ut,w_all_before,d,lamb,eta,largest_rho,m))
				else:
					w_all_prox = c@w_all_next + w_all_prox_before - c_tilde@w_all_before - eta*(self.gradient_partial(Ut,w_all_next,d,lamb,eta,largest_rho,m)-self.gradient_partial(Ut,w_all_before,d,lamb,eta,largest_rho,m))
				w_all_before = copy.deepcopy(w_all_next)
				w_all_next = self.all_extra_L1(Ut,d,w_all_prox,lamb,eta)
				bar = '='*int(10*i/iteration) + ("=" if i == 10 else ">")  + "."*(10-int(10*i/iteration)-1)    # プログレスバーの先頭の表示を工夫
				print(f"\r\033[K[{bar}] {(i+1)/iteration*100:.02f}% ({i+1}/{iteration})", end="")
				w_all_next_list.append(w_all_next[:,1:4])
				
				# print(w_all_before[0])
			if graph:
				times = range(len(average_error))
				plt.plot(times,average_error,label = "PG-EXTRA with Projective MC penalty")
			# plt.title("convergence over iteration")
			# plt.show()
			if animation:
				return average_error,w_all_next_list
			else:
				return average_error,np.mean(w_all_before,axis = 0)
		else:
			return None,None

	def partial_convexity_checker(self,rho,Ut):
		norm_list = []
		for ui in Ut:
			norm_list.append((np.linalg.norm(ui))**4)
		if rho > min(norm_list) or rho > 1:
			print(f"your formulation is not convex. must be smaller than {min(norm_list)} or 1. currently {rho}")
			return False
		else:
			return True
			


	def partial_eta_calculator(self,Ut,eta_input,c_tilde_min,lamb,rho):
		L = (1+lamb+rho)
		if L < 0:
			print(f"lamb + rho must be smaller than 1. currently {lamb + rho}")
		return eta_input*c_tilde_min/L

	def pg_extra_partial_mc_norm(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta_input,rho,iteration,c,w_all,decay,graph=True,animation=False):
		average_error = []
		# average_variance = []
		# average_convergence = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		w_all_prox_before = copy.deepcopy(w_all_prox)
		
		pre_c = decay*(1/((r_i+1)))*(c-np.eye(len(c)))
		c = pre_c + np.eye(len(c))*(1-r_i*decay/((r_i+1)))
		c_tilde = (1/2)*(np.eye(len(c))+c)
		c_tilde_min = min(LA.eig(c_tilde)[0])
		lip = max(LA.eig(Ut.T@Ut)[0])
		Ls = (1-rho)*lip+ rho
		eta = eta_input*c_tilde_min/Ls
		w_all_next_list = []
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			# average_convergence.append(LA.norm(w_all_before-w_all_next))
			# average_variance.append(np.var(w_all_before,axis = 1))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			if i == 0:
				w_all_prox = c@w_all_before-eta_input*(self.gradient_partial_norm(Ut,w_all_before,d,lamb,eta,rho,m))
			else:
				w_all_prox = c@w_all_next + w_all_prox_before - c_tilde@w_all_before - eta_input*(self.gradient_partial_norm(Ut,w_all_next,d,lamb,eta,rho,m)-self.gradient_partial_norm(Ut,w_all_before,d,lamb,eta,rho,m))
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,lamb,eta_input)
			
			w_all_next_list.append(w_all_next[:,1:4])
			
			# print(w_all_before[0])
		if graph:
			times = range(len(average_error))
			plt.plot(times,average_error,label = "PG-EXTRA with Projective MC penalty")
		# plt.title("convergence over iteration")
		# plt.show()
		if animation:
			return average_error,w_all_next_list
		else:
			return average_error,np.mean(w_all_before,axis = 0)
	
	def pg_extra_partial_mc_each_param(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta_input,rho,iteration,c,w_all,decay,graph=True,animation=False):
		average_error = []
		# average_variance = []
		# average_convergence = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		w_all_prox_before = copy.deepcopy(w_all_prox)
		
		pre_c = decay*(1/((r_i+1)))*(c-np.eye(len(c)))
		c = pre_c + np.eye(len(c))*(1-r_i*decay/((r_i+1)))
		c_tilde = (1/2)*(np.eye(len(c))+c)
		c_tilde_min = min(LA.eig(c_tilde)[0])
		lip = max(LA.eig(Ut.T@Ut)[0])
		Ls = (1-rho)*lip+ rho
		eta = (min(eta_input*c_tilde_min/Ls)).real
		w_all_next_list = []
		# print(eta)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			# average_convergence.append(LA.norm(w_all_before-w_all_next))
			# average_variance.append(np.var(w_all_before,axis = 1))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			if i == 0:
				# print("eta",eta)
				# print(c@w_all_before)
				# print(self.gradient_partial_each_param(Ut,w_all_before,d,lamb,eta,rho,m))
				w_all_prox = c@w_all_before-eta*(self.gradient_partial_each_param(Ut,w_all_before,d,lamb,eta,rho,m))
			else:
				w_all_prox = c@w_all_next + w_all_prox_before - c_tilde@w_all_before - eta*(self.gradient_partial_each_param(Ut,w_all_next,d,lamb,eta,rho,m)-self.gradient_partial_each_param(Ut,w_all_before,d,lamb,eta,rho,m))
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1_each_param(Ut,d,w_all_prox,lamb,eta)
			if i % 1000 == 0:
				print(f"iteration: {i}")
			w_all_next_list.append(w_all_next[:,1:4])
			
			# print(w_all_before[0])
		if graph:
			times = range(len(average_error))
			plt.plot(times,average_error,label = "PG-EXTRA with Projective MC penalty (Individual Parameter)")
		# plt.title("convergence over iteration")
		# plt.show()
		if animation:
			return average_error,w_all_next_list
		else:
			return average_error,np.mean(w_all_before,axis = 0)
	
	def pg_extra_partial_mc_each_param_norm(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta_input,rho,iteration,c,w_all,decay,graph=True,animation=False):
		average_error = []
		# average_variance = []
		# average_convergence = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		w_all_prox_before = copy.deepcopy(w_all_prox)
		
		pre_c = decay*(1/((r_i+1)))*(c-np.eye(len(c)))
		c = pre_c + np.eye(len(c))*(1-r_i*decay/((r_i+1)))
		c_tilde = (1/2)*(np.eye(len(c))+c)
		c_tilde_min = min(LA.eig(c_tilde)[0])
		lip = max(LA.eig(Ut.T@Ut)[0])
		Ls = (1-rho)*lip+ rho
		eta = min(eta_input*c_tilde_min/Ls)
		w_all_next_list = []
		# print(eta)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			# average_convergence.append(LA.norm(w_all_before-w_all_next))
			# average_variance.append(np.var(w_all_before,axis = 1))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			if i == 0:
				# print("eta",eta)
				# print(c@w_all_before)
				# print(self.gradient_partial_each_param(Ut,w_all_before,d,lamb,eta,rho,m))
				w_all_prox = c@w_all_before-eta*(self.gradient_partial_norm_each_param(Ut,w_all_before,d,lamb,eta,rho,m))
			else:
				w_all_prox = c@w_all_next + w_all_prox_before - c_tilde@w_all_before - eta*(self.gradient_partial_norm_each_param(Ut,w_all_next,d,lamb,eta,rho,m)-self.gradient_partial_norm_each_param(Ut,w_all_before,d,lamb,eta,rho,m))
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1_each_param(Ut,d,w_all_prox,lamb,eta)
			if i % 1000 == 0:
				print(f"iteration: {i}")
			w_all_next_list.append(w_all_next[:,1:4])
			
			# print(w_all_before[0])
		if graph:
			times = range(len(average_error))
			plt.plot(times,average_error,label = "PG-EXTRA with Projective MC penalty (Individual Parameter)")
		# plt.title("convergence over iteration")
		# plt.show()
		if animation:
			return average_error,w_all_next_list
		else:
			return average_error,np.mean(w_all_before,axis = 0)
	
	def pg_extra_mc_nonconvex_twin(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		# average_variance = []
		# average_convergence = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		w_all_prox_before = copy.deepcopy(w_all_prox)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		c_tilde_min = min(LA.eig(c_tilde)[0])
		lip = max(LA.eig(Ut.T@Ut)[0])
		Ls = (1-lamb)*lip+ rho
		eta = 2*c_tilde_min/Ls
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			# average_convergence.append(LA.norm(w_all_before-w_all_next))
			# average_variance.append(np.var(w_all_before,axis = 1))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			if i == 0:
				w_all_prox = c@w_all_before-eta*(self.gradient_mc(Ut,w_all_before,d,lamb,eta,rho,m))
			else:
				w_all_prox = c@w_all_next + w_all_prox_before - c_tilde@w_all_before - eta*(self.gradient_mc(Ut,w_all_next,d,lamb,eta,rho,m)-self.gradient_mc(Ut,w_all_before,d,lamb,eta,rho,m))
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,eta,lamb)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		plt.plot(times,average_error,label = "PG-EXTRA with MC penalty")
		# plt.title("convergence over iteration")
		# plt.show()
		return np.mean(w_all_before,axis = 0)

	def pg_extra_dig_mc_soft(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		# average_variance = []
		# average_convergence = []
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		w_all_prox_before = copy.deepcopy(w_all_prox)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		c_tilde_min = min(LA.eig(c_tilde)[0])
		lip = max(LA.eig(Ut.T@Ut)[0])
		Ls = (1-lamb)*lip+ rho
		print("step size must be smaller than", 2*c_tilde_min/Ls)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			# average_convergence.append(LA.norm(w_all_before-w_all_next))
			# average_variance.append(np.var(w_all_before,axis = 1))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			if i == 0:
				w_all_prox = c@w_all_before-eta*c@(self.gradient_soft(Ut,w_all_before,d,lamb,eta,rho,m))
			else:
				w_all_prox = c@w_all_next + w_all_prox_before - c_tilde@w_all_before - eta*c@(self.gradient_soft(Ut,w_all_next,d,lamb,eta,rho,m)-self.gradient_soft(Ut,w_all_before,d,lamb,eta,rho,m))
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,eta,lamb)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		plt.plot(times,average_error)
		# plt.title("convergence over iteration")
		# plt.show()
		return average_error

	def pg_extra_mc_soft_nonconvex(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_k = copy.deepcopy(w_all)
		w_all_k1 = copy.deepcopy(w_all)
		w_all_k2 = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		w_all_prox_before = copy.deepcopy(w_all_prox)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_k,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			if i == 0:
				w_all_prox = c@w_all_k-eta*(self.gradient_soft_nonconvex(Ut,w_all_k1,d,lamb,eta,rho))
			else:
				w_all_prox = c@w_all_k1 + w_all_prox_before - c_tilde@w_all_k - eta*(self.gradient_soft_nonconvex(Ut,w_all_k1,d,lamb,eta,rho)-self.gradient_soft_nonconvex(Ut,w_all_k,d,lamb,eta,rho))
			w_all_k = copy.deepcopy(w_all_k1)
			w_all_k2 = self.all_extra_L1(Ut,d,w_all_prox,eta,lamb)
			w_all_k1 = copy.deepcopy(w_all_k2)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'PG-EXTRA with MC penalty')
		return np.mean(w_all_k1,axis = 0)

	def pg_extra_mc_L2_soft(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		w_all_prox_before = copy.deepcopy(w_all_prox)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			if i == 0:
				w_all_prox = c@w_all_before-eta*(self.gradient_L2_soft(Ut,w_all_next,d,lamb,eta,rho))
			else:
				w_all_prox = c@w_all_next + w_all_prox_before - c_tilde@w_all_before - eta*(self.gradient_L2_soft(Ut,w_all_next,d,lamb,eta,rho)-self.gradient_L2_soft(Ut,w_all_before,d,lamb,eta,rho))
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,eta,lamb)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'extra mc L2')
		return np.mean(w_all_next,axis = 0)

	def extra(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_before = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_between = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		la, v = np.linalg.eig(c_tilde)
		s = min(la)
		step_size = 10**30
		for u in Ut:
			ut = np.reshape(u,(len(u),1)) 
			um = np.reshape(u,(1,len(u)))
			lip =abs(um@ut)
			lpz = 2*(s)/lip
			if lpz[0][0] < step_size:
				step_size = lpz[0][0]

		eta_this = step_size
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_between = copy.deepcopy(w_all_next)
			w_all_next = self.extra_share(Ut,d,c,c_tilde,w_all_next,w_all_before,eta_this,i)
			w_all_before = copy.deepcopy(w_all_between)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'EXTRA')
		return average_error,np.mean(w_all_next,axis = 0)

	def atc_extra(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_before = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_between = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		la, v = np.linalg.eig(c_tilde)
		s = min(la)
		eta_this = eta
		for u in Ut:
			ut = np.reshape(u,(len(u),1)) 
			um = np.reshape(u,(1,len(u)))
			lip =abs(um@ut)
			lpz = 2*(s)/lip
			if lpz[0][0] < eta_this:
				print(f"ATC EXTRA condition: eta {eta} must be smaller than {lpz}")
				# eta_this = lpz[0][0]
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_between = copy.deepcopy(w_all_next)
			w_all_next = self.atc_extra_share(Ut,d,c,c_tilde,w_all_next,w_all_before,eta_this,i)
			w_all_before = copy.deepcopy(w_all_between)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'ATC EXTRA')
		return np.mean(w_all_next,axis = 0)

	def atc_extra_tilde(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_before = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_between = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_between = copy.deepcopy(w_all_next)
			w_all_next = self.atc_extra_share_tilde(Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i)
			w_all_before = copy.deepcopy(w_all_between)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'ATC tilde EXTRA')
		return np.mean(w_all_next,axis = 0)
	
	def atc_extra_2(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_before = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_between = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_between = copy.deepcopy(w_all_next)
			w_all_next = self.atc_extra_share_2(Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i)
			w_all_before = copy.deepcopy(w_all_between)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'ATC 2 EXTRA')
		return np.mean(w_all_next,axis = 0)

	def dig(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_before = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_between = copy.deepcopy(w_all)
		y_all = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_between = copy.deepcopy(w_all_next)
			if i == 0:
				y_all = self.gradient(Ut,w_all_next,d)
				w_all_next = w_all_next
			else:
				w_all_next = c@w_all_before - eta*y_all 
				y_all = c@y_all + self.gradient(Ut,w_all_next,d)- self.gradient(Ut,w_all_before,d)
			w_all_before = copy.deepcopy(w_all_next)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'Dig')
		return np.mean(w_all_next,axis = 0)

	def atc_dig(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_before = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		y_all = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			if i == 0:
				y_all = self.gradient(Ut,w_all_next,d)
				w_all_next = w_all_next
			else:
				w_all_next = c@(w_all_before - eta*y_all)
				y_all = c@(y_all + self.gradient(Ut,w_all_next,d)- self.gradient(Ut,w_all_before,d))
			w_all_before = copy.deepcopy(w_all_next)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'atc Dig')
		return np.mean(w_all_next,axis = 0)

	def extra_new(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all_before = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_between = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_between = copy.deepcopy(w_all_next)
			w_all_next = self.extra_share_new(Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i)
			w_all_before = copy.deepcopy(w_all_between)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'extra new')
		return np.mean(w_all_next,axis = 0)
	
	def extra_share(self,Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i):	
		if i == 0:
			return c@w_all_next - eta*(self.gradient(Ut,w_all_next,d))
		else:
			return c@w_all_next + w_all_next - c_tilde@w_all_before - eta*(self.gradient(Ut,w_all_next,d)-self.gradient(Ut,w_all_before,d))
	
	def atc_extra_share(self,Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i):
		if i == 0:
			return c@(w_all_next - eta*(self.gradient(Ut,w_all_next,d)))
		else:
			return c@w_all_next + w_all_next - c_tilde@w_all_before - eta*c@(self.gradient(Ut,w_all_next,d)-self.gradient(Ut,w_all_before,d))
	
	def atc_extra_share_tilde(self,Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i):
		if i == 0:
			return c@(w_all_next - eta*(self.gradient(Ut,w_all_next,d)))
		else:
			return c@w_all_next + w_all_next - c_tilde@w_all_before - eta*c_tilde@(self.gradient(Ut,w_all_next,d)-self.gradient(Ut,w_all_before,d))

	def atc_extra_share_2(self,Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i):
		if i == 0:
			return c@(w_all_next - eta*(self.gradient(Ut,w_all_next,d)))
		else:
			return c@w_all_next + w_all_next - c_tilde@w_all_before - eta*(c@self.gradient(Ut,w_all_next,d)-c_tilde@self.gradient(Ut,w_all_before,d))

	def dig_one(self,Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i):
		if i == 0:
			return c@(w_all_next - eta*(self.gradient(Ut,w_all_next,d)))
		else:
			return c@w_all_next + w_all_next - c_tilde@w_all_before - eta*(c@self.gradient(Ut,w_all_next,d)-c_tilde@self.gradient(Ut,w_all_before,d))

	def extra_mixing_matrix_l1_share(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox,eta,i,c_2,I,mu):
		if i == 0:
			return c@w_all_next - eta*(self.gradient(Ut,w_all_next,d)+ mu*(I - c_2)@w_all_next)
		else:
			return c@w_all_next + w_all_prox - c_tilde@w_all_before - eta*(self.gradient(Ut,w_all_next,d)+mu*(I - c_2)@w_all_next-self.gradient(Ut,w_all_before,d)-mu*(I - c_2)@w_all_before)
	
	def extra_mixing_matrix_l1_share_CPT_with_mixing_matrix(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox,eta,i,c_2,I,mu):
		if i == 0:
			return c@w_all_next - eta*(self.gradient(Ut,w_all_next,d)+ mu*(I-c_tilde)@w_all_next)
		else:
			return c@w_all_next + w_all_prox - c_tilde@w_all_before - eta*(self.gradient(Ut,w_all_next,d)+mu*(I - c_tilde)@w_all_next-self.gradient(Ut,w_all_before,d)-mu*(I - c_tilde)@w_all_before)
	
	def extra_l1_consensus_violation_share(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox,eta,i,I,rho,m,c_2,mu):
		if i == 0:
			return c@w_all_next - eta*(self.gradient(Ut,w_all_next,d) + mu*(I -c_2)@w_all_next)
		else:
			return c@w_all_next + w_all_prox - c_tilde@w_all_before - eta*(self.gradient(Ut,w_all_next,d) + mu*(I -c_2)@w_all_next - self.gradient(Ut,w_all_before,d) - mu*(I -c_2)@w_all_before)
	
	def extra_l1_share(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox,eta,i):
		if i == 0:
			return c@w_all_next - eta*(self.gradient(Ut,w_all_next,d))
		else:
			return c@w_all_next + w_all_prox - c_tilde@w_all_before - eta*(self.gradient(Ut,w_all_next,d)-self.gradient(Ut,w_all_before,d))
	
	def extra_l1_share_norm(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox,eta,i):
		if i == 0:
			return c@w_all_next - eta*(self.gradient_norm(Ut,w_all_next,d))
		else:
			return c@w_all_next + w_all_prox - c_tilde@w_all_before - eta*(self.gradient_norm(Ut,w_all_next,d)-self.gradient_norm(Ut,w_all_before,d))

	def extra_l1_share_step_size(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox,eta_mat,i):
		if i == 0:
			return c@w_all_next - eta_mat@(self.gradient(Ut,w_all_next,d))
		else:
			return c@w_all_next + w_all_prox - c_tilde@w_all_before - c@eta_mat@(self.gradient(Ut,w_all_next,d)-self.gradient(Ut,w_all_before,d))
	
	def atc_extra_l1_share(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox,eta,i):
		if i == 0:
			return c@w_all_next - eta*(self.gradient(Ut,w_all_next,d))
		else:
			return c@w_all_next + w_all_prox - c_tilde@w_all_before - eta*c@(self.gradient(Ut,w_all_next,d)-self.gradient(Ut,w_all_before,d))

	def atc_extra_l1_share_2(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_before_before,w_all_prox,eta,i):
		if i == 0:
			return c@w_all_next - eta*(self.gradient(Ut,w_all_next,d))
		if i == 1:
			return c@w_all_next + w_all_prox - c_tilde@w_all_before - eta*c@(self.gradient(Ut,w_all_next,d)-self.gradient(Ut,w_all_before,d))
		else:
			return c@w_all_next + w_all_prox - c_tilde@w_all_before - eta*c@(self.gradient(Ut,w_all_next,d)-self.gradient(Ut,w_all_before,d)) + eta*c_tilde@(self.gradient(Ut,w_all_before,d)-self.gradient(Ut,w_all_before_before,d))
	
	def atc_extra_l1_share_3(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox,eta,i):
		if i == 0:
			return c@w_all_next - eta*(self.gradient(Ut,w_all_next,d))
		else:
			return c@w_all_next + w_all_prox - c_tilde@w_all_before - eta*c_tilde@(self.gradient(Ut,w_all_next,d)-self.gradient(Ut,w_all_before,d))
	
	def prox_dgd_update(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox,lamb,eta,i):
		return c@w_all_next - eta*(self.gradient(Ut,w_all_next,d))
	
	def prox_dgd_consensus_violation_update(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox,lamb,eta,i,I,c_2):
		return c@w_all_next - eta*(self.gradient(Ut,w_all_next,d) + (I-c_2)@w_all_next)
		
	def extra_share_new(self,Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i):
		if i == 0:
			return c@w_all_next - eta*(self.gradient_new(Ut,w_all_next,d))
		else:
			return c@w_all_next + w_all_next - c_tilde@w_all_before - eta*(self.gradient_new(Ut,w_all_next,d)-self.gradient_new(Ut,w_all_before,d))

	def extra_share_soft(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,lamb,rho,i):
		if i == 0:
			return c@w_all_next - eta*(self.gradient_soft(Ut,w_all_next,d,lamb,eta,rho))
		else:
			return c@w_all_next + w_all_prox_before - c_tilde@w_all_before - eta*(self.gradient_soft(Ut,w_all_next,d,lamb,eta,rho)-self.gradient_soft(Ut,w_all_before,d,lamb,eta,rho))

	def new_extra_share(self,Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i):
		U = Ut.T
		if i == 0:
			return ((c@w_all_next).T - eta*(U@(Ut@w_all_next.T-d))).T
		else:
			return ((c@w_all_next).T + w_all_next.T - (c_tilde@w_all_before).T - eta*(U@(Ut@w_all_next.T-d)-U@(Ut@w_all_before.T-d))).T

	def pg_extra_l1(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,mu,decay,graph=True,animation=False):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		# c = (1/(r_i+1))*c
		# c_tilde = (1/2)*(np.eye(len(c))+c)
		I,c,c_2,c_tilde,eta = self.extra_setup(c,r_i,Ut,mu,1,rho,decay)
		eta = self.pg_extra_step_size(Ut,c_tilde)
		w_all_next_list = []
		# print("l1 eta",eta)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_share(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,lamb,eta)
			bar = '='*int(10*i/iteration) + ("=" if i == 10 else ">")  + "."*(10-int(10*i/iteration)-1)    # プログレスバーの先頭の表示を工夫
			print(f"\r\033[K[{bar}] {(i+1)/iteration*100:.02f}% ({i+1}/{iteration})", end="")
			w_all_next_list.append(w_all_next[:,1:4])			
		if graph:
			times = range(len(average_error))
			plt.plot(times,average_error,label = 'PG-EXTRA with ' + r"$\ell_1$"+' penalty')
		if animation:
			return average_error,w_all_next_list
		else:
			return average_error,np.mean(w_all_next,axis = 0)
	
	def pg_extra_l1_each_param(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,mu,decay,graph=True,animation=False):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		# c = (1/(r_i+1))*c
		# c_tilde = (1/2)*(np.eye(len(c))+c)
		I,c,c_2,c_tilde,eta = self.extra_setup(c,r_i,Ut,mu,1,rho,decay)
		eta = self.pg_extra_step_size(Ut,c_tilde)
		w_all_next_list = []
		print("l1 eta",eta)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_share(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1_each_param(Ut,d,w_all_prox,lamb,eta)
			if i %300 == 0:
				print("iteration:",i)
			w_all_next_list.append(w_all_next[:,1:4])
			
		if graph:
			times = range(len(average_error))
			plt.plot(times,average_error,label = 'PG-EXTRA with ' + r"$\ell_1$"+' penalty Individual Parameter')
		if animation:
			return average_error,w_all_next_list
		else:
			return average_error,np.mean(w_all_next,axis = 0)
	
	def pg_extra_l1_norm(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,mu,decay,graph=True,animation=False):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		# c = (1/(r_i+1))*c
		# c_tilde = (1/2)*(np.eye(len(c))+c)
		I,c,c_2,c_tilde,eta = self.extra_setup(c,r_i,Ut,mu,1,rho,decay)
		eta = self.pg_extra_step_size(Ut,c_tilde)
		w_all_next_list = []
		print("l1 eta",eta)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_share_norm(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,lamb,eta)
			if i %300 == 0:
				print("iteration:",i)
			w_all_next_list.append(w_all_next[:,1:4])
			
		if graph:
			times = range(len(average_error))
			plt.plot(times,average_error,label = 'PG-EXTRA with ' + r"$\ell_1$"+' penalty')
		if animation:
			return average_error,w_all_next_list
		else:
			return average_error,np.mean(w_all_next,axis = 0)
	
	def pg_extra_mixing_matrix_l1(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,mu,decay):
		average_error = []
		neighbor_consensus_error=[]
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		I=np.eye(len(c))
		c_2 = (1/(r_i))*(c-I)
		pre_c = decay*(1/((r_i+1)))*(c-np.eye(len(c)))
		c = pre_c + np.eye(len(c))*(1-r_i*decay/((r_i+1)))
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_next_average = np.reshape(np.mean(w_all_next,axis = 0),(len(w_star),1))
			L2_consensus = np.linalg.norm(w_all_next_average, ord=2)
			neighbor_consensus_error.append(self.error_consensus(w_all_next,N,L2_consensus,m,c_2))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_mixing_matrix_l1_share(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i,c_2,I,mu)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,lamb,eta)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		# plt.plot(times,average_error,label = 'PG-EXTRA with ' + r"$\ell_1$"+' penalty')
		plt.plot(times,neighbor_consensus_error,label = 'PG-EXTRA with ' + r"$\ell_1$"+' penalty, mm param = ' + f"{decay} "+ "mu = "+ f"{mu}")
		return np.mean(w_all_next,axis = 0)

	def pg_extra_mixing_matrix_l1_consensus_graph(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,mu,decay):
		average_error = []
		error = []
		neighbor_consensus_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		I=np.eye(len(c))
		c_2 = (1/(r_i))*(c-I)
		pre_c = decay*(1/((r_i+1)))*(c-np.eye(len(c)))
		c = pre_c + np.eye(len(c))*(1-r_i*decay/((r_i+1)))
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			w_average = np.mean(w_all_next,axis = 0)
			w_average = np.reshape(w_average,(len(w_average),1))
			L2_2 = np.linalg.norm(w_average,ord=2)
			w_all_next_average = np.reshape(np.mean(w_all_next,axis = 0),(len(w_star),1))
			L2_consensus = np.linalg.norm(w_all_next_average, ord=2)
			average_error.append(self.error_distributed(w_all_next,w_average,N,L2_2,m))
			error.append(self.error_distributed(w_all_next,w_star,N,L2,m))
			neighbor_consensus_error.append(self.error_consensus(w_all_next,N,L2_consensus,m,c_2))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_mixing_matrix_l1_share(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i,c_2,I,mu)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,lamb,eta)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'consensus error, mm param = '+f"{decay}, mu = "+f"{mu}")
		plt.plot(times,neighbor_consensus_error,label = 'Neighbor consensus mm param = '+f"{decay}, mu = "+f"{mu}")
		return np.mean(w_all_next,axis = 0)

	def pg_extra_mixing_matrix_l1_consensus_graph_neighbor_mixing_matrix(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,mu,decay):
		average_error = []
		error = []
		neighbor_consensus_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		I=np.eye(len(c))
		c_2 = (1/(r_i))*(c-I)
		pre_c = decay*(1/((r_i+1)))*(c-np.eye(len(c)))
		c = pre_c + np.eye(len(c))*(1-r_i*decay/((r_i+1)))
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			w_average = np.mean(w_all_next,axis = 0)
			w_average = np.reshape(w_average,(len(w_average),1))
			L2_2 = np.linalg.norm(w_average,ord=2)
			w_all_next_average = np.reshape(np.mean(w_all_next,axis = 0),(len(w_star),1))
			L2_consensus = np.linalg.norm(w_all_next_average, ord=2)
			average_error.append(self.error_distributed(w_all_next,w_average,N,L2_2,m))
			error.append(self.error_distributed(w_all_next,w_star,N,L2,m))
			neighbor_consensus_error.append(self.error_consensus(w_all_next,N,L2_consensus,m,c_2))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_mixing_matrix_l1_share_CPT_with_mixing_matrix(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i,c_2,I,mu)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,lamb,eta)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'consensus error, mm param = '+f"{decay}, mu = "+f"{mu}")
		plt.plot(times,neighbor_consensus_error,label = 'Neighbor consensus mm param = '+f"{decay}, mu = "+f"{mu}")
		return np.mean(w_all_next,axis = 0)

	def pad(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,epsilon,iteration,c,w_all,alpha):
		average_error = []
		w_all = copy.deepcopy(w_all)
		prox = copy.deepcopy(w_all)
		pi = copy.deepcopy(w_all)
		z = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		I = np.eye(len(c))
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all,w_star,N,L2,m))
			prox = w_all - eta*(self.gradient(Ut,w_all,d)+alpha*((I-c)@w_all-z)+ pi)
			w_all = self.all_extra_L1(Ut,d,prox,eta,lamb)
			z = (1/(alpha+(1/epsilon)))*(pi + alpha*(I-c)@w_all)
			pi = pi + alpha*((I-c)@w_all-z)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'PAD')
		return np.mean(w_all,axis = 0)
	
	def dfbbs(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		prox = copy.deepcopy(w_all)
		y = copy.deepcopy(w_all)
		y_before = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		I = np.eye(len(c))
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all,w_star,N,L2,m))
			prox = w_all - eta*self.gradient(Ut,w_all,d)+eta*(2*y-y_before)
			w_all = self.all_extra_L1(Ut,d,prox,eta,lamb)
			y_before = copy.deepcopy(y)
			y = y - (1/eta)*(I-c_tilde)@w_all
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'DFBBS')
		return np.mean(w_all,axis = 0)
	
	def dista(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,q):
		average_error = []
		w_all_dista =copy.deepcopy(w_all)
		w_all_average =copy.deepcopy(w_all)
		prox =copy.deepcopy(w_all)
		c = (1/(r_i))*(c-np.eye(len(c)))
		alpha = q*lamb/m
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_dista,w_star,N,L2,m))
			w_all_average = c@w_all_dista
			prox = (1-q)*c@w_all_average + q*(w_all_dista - eta*self.gradient(Ut,w_all_dista,d))
			w_all_dista = self.all_extra_L1(Ut,d,prox,alpha,eta)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'DISTA')
		return np.mean(w_all_dista,axis = 0)

	def pg_extra_l1_step_size(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		eta_matrix = copy.deepcopy(c)
		la, v = np.linalg.eig(c_tilde)
		s = min(la)
		for i in range(len(Ut)):
			u = Ut[i]
			ut = np.reshape(u,(len(u),1)) 
			um = np.reshape(u,(1,len(u)))
			lip =abs(um@ut)+lamb
			lpz = 2/lip
			eta_matrix[i][i] = eta*lpz
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_share_step_size(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta_matrix,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_step_size_L1(Ut,d,w_all_prox,lamb,eta_matrix)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'PG-EXTRA_step_size with ' + r"$\ell_1$"+' penalty')
		return np.mean(w_all_next,axis = 0)
	
	def pg_extra_l1_projection(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.extra_l1_share(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.l1_projection(Ut,d,w_all_prox,lamb,1)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'Projection-EXTRA with ' + r"$\ell_1$"+' penalty')
		return np.mean(w_all_next,axis = 0)
	
	def atc_pg_extra_l1_projection(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.atc_extra_l1_share(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.l1_projection(Ut,d,w_all_prox,lamb,1)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'ATC Projection-EXTRA with ' + r"$\ell_1$"+' penalty')
		return np.mean(w_all_next,axis = 0)
	
	def FDDA(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_1_t = copy.deepcopy(w_all)
		w_1_t1 = copy.deepcopy(w_all)
		h_t = self.gradient(Ut,w_1_t,d)
		s_t = self.gradient(Ut,w_1_t,d)
		s_t1 = self.gradient(Ut,w_1_t,d)
		c = (1/(r_i+1))*c
		A_h = 0
		for i in range(1,iteration+1):
			average_error.append(self.error_distributed(w_1_t,w_star,N,L2,m))
			A_h -= eta*h_t
			w_1_t = copy.deepcopy(w_1_t1)
			w_1_t1 = self.l1_projection(Ut,d,A_h,lamb,1)
			s_t = copy.deepcopy(s_t1)
			s_t1 = c@s_t + self.gradient(Ut,w_1_t1,d) - self.gradient(Ut,w_1_t,d)
			h_t = c@h_t + s_t1 - s_t
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'FDDA with ' + r"$\ell_1$"+' penalty')
		return np.mean(w_1_t,axis = 0)

	def ADDA(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_1_t = copy.deepcopy(w_all)
		w_1_t1 = copy.deepcopy(w_all)
		w_2 = copy.deepcopy(w_all)
		w_hat_t1 = copy.deepcopy(w_all)
		s_t = self.gradient(Ut,w_1_t,d)
		s_t1 = self.gradient(Ut,w_1_t,d)
		c = (1/(r_i+1))*c
		eta_t = 0
		eta_t1 = eta
		A_t = 0
		A_t1 = 0
		A_s = copy.deepcopy(s_t1)
		for i in range(1,iteration+1):
			A_t += eta_t
			A_t1 += eta_t1
			average_error.append(self.error_distributed(w_1_t1,w_star,N,L2,m))
			w_1_t = copy.deepcopy(w_1_t1)
			w_1_t1 = (A_t/A_t1)*c@w_2 + (eta_t1/A_t1)*w_hat_t1
			s_t = copy.deepcopy(s_t1)
			s_t1 = c@s_t + self.gradient(Ut,w_1_t1,d) - self.gradient(Ut,w_1_t,d)
			A_s -= eta_t1*s_t
			w_hat_t1 = self.l1_projection(Ut,d,A_s,lamb,1)
			w_2 = (A_t/A_t1)*c@w_2 + (eta_t1/A_t1)*w_hat_t1
			if i %100 == 0:
				print("iteration:",i)
			eta_t = eta*i
			eta_t1 = eta*(i + 1)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'ADDA with ' + r"$\ell_1$"+' Penalty')
		return np.mean(w_1_t,axis = 0)

	def distributed_proximal_gradient_algorithm(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,graph,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		p = copy.deepcopy(w_all)
		s = copy.deepcopy(w_all)
		gamma = (eta*m/(0.5*r_i*m*r_i))**0.5
		G = -gamma*gamma/(2*gamma)
		G_middle = r_i*(-G)*np.eye(len(graph))
		graph = G*(graph - np.eye(len(graph))) + G_middle
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all,w_star,N,L2,m))
			w_all = w_all - eta*(self.gradient(Ut,w_all,d) + p + s)
			w_all = self.all_extra_L1(Ut,d,w_all,lamb,eta)
			s = graph@w_all
			p = p + s
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'DPGA with ' + r"$\ell_1$"+' Penalty')
		return np.mean(w_all,axis = 0)
	
	def distributed_proximal_gradient_algorithm_consensus_violation(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,graph,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		p = copy.deepcopy(w_all)
		s = copy.deepcopy(w_all)
		I = np.eye(len(graph))
		c_2 = (1/r_i)*(graph - I)
		eigenvalue = LA.eig(I - c_2)[0]
		sigma = np.sort(eigenvalue)
		print(sigma)
		exit()
		gamma = (eta*m/(0.5*r_i*m*r_i))**0.5
		G = -gamma*gamma/(2*gamma)
		G_middle = r_i*(-G)*np.eye(len(graph))
		graph = G*(graph - np.eye(len(graph))) + G_middle
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all,w_star,N,L2,m))
			w_all = w_all - eta*(self.gradient(Ut,w_all,d) + (I - c_2)@w_all + p + s)
			w_all = self.all_extra_mc(Ut,d,w_all,lamb,eta,rho)
			s = graph@w_all
			p = p + s
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'DPGA with consensus violation and MC Penalty')
		return np.mean(w_all,axis = 0)

	def distributed_proximal_gradient_algorithm_consensus_violation_consensus_graph(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,graph,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		p = copy.deepcopy(w_all)
		s = copy.deepcopy(w_all)
		I = np.eye(len(graph))
		c_2 = (1/r_i)*(graph - I)
		gamma = (eta*m/(0.5*r_i*m*r_i))**0.5
		G = -gamma*gamma/(2*gamma)
		G_middle = r_i*(-G)*np.eye(len(graph))
		graph = G*(graph - np.eye(len(graph))) + G_middle
		for i in range(iteration):
			w_average = np.mean(w_all,axis = 0)
			w_average = np.reshape(w_average,(len(w_average),1))
			L2_2 = np.linalg.norm(w_average,ord=2)
			average_error.append(self.error_distributed(w_all,w_average,N,L2_2,m))
			w_all = w_all - eta*(self.gradient(Ut,w_all,d) + (I - c_2)@w_all + p + s)
			w_all = self.all_extra_mc(Ut,d,w_all,lamb,eta,rho)
			s = graph@w_all
			p = p + s
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = r'DPGA with consensus violation and MC Penalty')
		return np.mean(w_all,axis = 0)

	def distributed_proximal_gradient_algorithm_firm(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,graph,w_all):
		B = (rho/lamb)**0.5
		self.distributed_convexity_checker(B,lamb,Ut,len(w_all.T))
		average_error = []
		w_all = copy.deepcopy(w_all)
		p = copy.deepcopy(w_all)
		s = copy.deepcopy(w_all)
		gamma = (eta*m/(0.5*r_i*m*r_i))**0.5
		G = -gamma*gamma/(2*gamma)
		G_middle = r_i*(-G)*np.eye(len(graph))
		graph = G*(graph - np.eye(len(graph))) + G_middle
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all,w_star,N,L2,m))
			w_all = w_all - eta*(self.gradient(Ut,w_all,d) + p + s)
			w_all = self.all_extra_mc(Ut,d,w_all,lamb,eta,rho)
			s = graph@w_all
			p = p + s
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'DPGA with MC penalty')
		return np.mean(w_all,axis = 0)

	def distributed_proximal_gradient_algorithm_firm_consensus_graph(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,graph,w_all):
		B = (rho/lamb)**0.5
		self.distributed_convexity_checker(B,lamb,Ut,len(w_all.T))
		average_error = []
		w_all = copy.deepcopy(w_all)
		p = copy.deepcopy(w_all)
		s = copy.deepcopy(w_all)
		gamma = (eta*m/(0.5*r_i*m*r_i))**0.5
		G = -gamma*gamma/(2*gamma)
		G_middle = r_i*(-G)*np.eye(len(graph))
		graph = G*(graph - np.eye(len(graph))) + G_middle
		for i in range(iteration):
			w_average = np.mean(w_all,axis = 0)
			w_average = np.reshape(w_average,(len(w_average),1))
			L2_2 = np.linalg.norm(w_average,ord=2)
			average_error.append(self.error_distributed(w_all,w_average,N,L2_2,m))
			w_all = w_all - eta*(self.gradient(Ut,w_all,d) + p + s)
			w_all = self.all_extra_mc(Ut,d,w_all,lamb,eta,rho)
			s = graph@w_all
			p = p + s
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = r'DPGA with MC penalty')
		return np.mean(w_all,axis = 0)

	def distributed_proximal_gradient_algorithm_approximate_MC(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,graph,w_all):
		B = (rho/lamb)**0.5
		self.distributed_convexity_checker(B,lamb,Ut,len(w_all.T))
		average_error = []
		w_all = copy.deepcopy(w_all)
		p = copy.deepcopy(w_all)
		s = copy.deepcopy(w_all)
		gamma = (eta*m/(0.5*r_i*m*r_i))**0.5
		G = -gamma*gamma/(2*gamma)
		G_middle = r_i*(-G)*np.eye(len(graph))
		graph = G*(graph - np.eye(len(graph))) + G_middle
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all,w_star,N,L2,m))
			w_all = w_all - eta*(self.gradient_soft(Ut,w_all,d,lamb,eta,rho,m) + p + s)
			w_all = self.all_extra_L1(Ut,d,w_all,lamb,eta)
			s = graph@w_all
			p = p + s
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'DPGA with Approximate MC penalty')
		return np.mean(w_all,axis = 0)
	
	def APM(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		y = copy.deepcopy(w_all)
		s = copy.deepcopy(w_all)
		theta = 0.5
		theta_before = 1
		L = 0
		for i in range(len(Ut)):
			u = Ut[i]
			ut = np.reshape(u,(len(u),1)) 
			um = np.reshape(u,(1,len(u)))
			lip =abs(um@ut)
			if lip[0][0] > L:
				L = lip[0][0]
		eigenvalue = LA.eig(c_tilde)[0]
		sigma = np.sort(eigenvalue)[-2]
		beta = (max([lamb,L])/(1-sigma)**0.5)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			y = w_all + (theta*(1-theta_before)/theta_before)*(w_all-w_all_before)
			s = self.gradient(Ut,y,d) + (beta/theta)*(y-c_tilde@y)
			S = y-(1/(L+beta/theta))*s
			w_all_before = copy.deepcopy(w_all)
			w_all = self.all_extra_L1(Ut,d,S,lamb,eta)
			theta_before = copy.deepcopy(theta)
			theta = theta/(theta + 1)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'APM with ' + r"$\ell_1$"+' penalty')
		return np.mean(w_all,axis = 0)
	
	def atc_pg_extra_l1(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.atc_extra_l1_share(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,lamb,eta)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'ATC PG-EXTRA (1) with ' + r"$\ell_1$"+' penalty')
		return np.mean(w_all_next,axis = 0)
	
	def atc_pg_extra_l1_2(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.atc_extra_l1_share(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before_before = copy.deepcopy(w_all_before)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,lamb,eta)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'atc PG-EXTRA 2 with ' + r"$\ell_1$"+' penalty')
		return np.mean(w_all_next,axis = 0)
	
	def atc_pg_extra_l1_3(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.atc_extra_l1_share_2(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,lamb,eta)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'ATC PG-EXTRA (2) with ' + r"$\ell_1$"+' penalty')
		return np.mean(w_all_next,axis = 0)

	def prox_dgd(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all,graph_plot=True):
		average_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)
		lip = max(LA.eig(Ut.T@Ut)[0])
		Ls = lip
		c_tilde_min = min(LA.eig(c_tilde)[0])
		eta = 0.4*(c_tilde_min+1)/Ls
		print("proxdgd eta",eta)
		rho = 1/lamb
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.prox_dgd_update(Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox_before,lamb,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc(Ut,d,w_all_prox,lamb,eta,rho)
			bar = '='*int(100*i/iteration) + ("=" if i == 100 else ">")  + "."*(100-int(100*i/iteration)-1)    # プログレスバーの先頭の表示を工夫
			print(f"\r\033[K[{bar}] {(i+1)/iteration*100:.02f}% ({i+1}/{iteration})", end="")
		times = range(len(average_error))
		if graph_plot:
			plt.plot(times,average_error,label = 'Prox DGD with MC penalty')
		return average_error

	def prox_dgd_consensus_graph(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		consensus_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c_1 = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c_1)
		for i in range(iteration):
			w_all_before_average = np.reshape(np.mean(w_all_before,axis = 0),(len(w_star),1))
			L2_consensus = np.linalg.norm(w_all_before_average, ord=2)
			consensus_error.append(self.error_distributed(w_all_before,w_all_before_average,N,L2_consensus,m))
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox_before = copy.deepcopy(w_all_prox)
			w_all_prox = self.prox_dgd_update(Ut,d,c_1,c_tilde,w_all_next,w_all_before,w_all_prox_before,lamb,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc(Ut,d,w_all_prox,lamb,eta,rho)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		# plt.plot(times,average_error,label = 'prox dgd with mc penalty')
		plt.plot(times,consensus_error,label = 'Prox DGD with MC penalty consensus violation')
		return average_error

	def prox_dgd_consensus_violation_consensus_graph(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
		consensus_error = []
		w_all = copy.deepcopy(w_all)
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		I = np.eye(len(c))
		c_2 = (1/(r_i))*(c-I)
		c_1 = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c_1)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_next,w_star,N,L2,m))
			w_all_next_average = np.reshape(np.mean(w_all_next,axis = 0),(len(w_star),1))
			L2_consensus = np.linalg.norm(w_all_next_average, ord=2)
			consensus_error.append(self.error_distributed(w_all_next,w_all_next_average,N,L2_consensus,m))
			w_all_prox = self.prox_dgd_consensus_violation_update(Ut,d,c_1,c_tilde,w_all_next,w_all_before,w_all_before,lamb,eta,i,I,c_2)
			w_all_next = self.all_extra_mc(Ut,d,w_all_prox,lamb,eta,rho)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		# plt.plot(times,average_error,label = 'prox dgd with mc penalty')
		plt.plot(times,consensus_error,label = 'prox dgd with mc penalty consensus violation')
		return average_error

	def gradient(self,Ut,w_now,d):
		w = copy.deepcopy(w_now)
		gradient = copy.deepcopy(w)
		for i in range(len(Ut)):
			gradient[i] = self.one_gradient(Ut[i],d[i],w[i])
		return gradient

	def gradient_norm(self,Ut,w_now,d):
		gradient_now = self.gradient(Ut,w_now,d)
		gradient_norm = np.zeros(gradient_now.shape)
		for i in range(len(gradient_norm)):
			gradient_norm[i] = gradient_now[i]/np.linalg.norm(gradient_now[i])
		return gradient_norm
	
	def negative_gradient(self,Ut,w_now,d):
		w = copy.deepcopy(w_now)
		gradient = copy.deepcopy(w)
		for i in range(len(Ut)):
			gradient[i] = self.one_negative_gradient(Ut[i],d[i],w[i])
		return gradient
	
	def mc_prox_dgd_gradient(self,Ut,w_now,d,lamb,eta,B):
		w = copy.deepcopy(w_now)
		gradient = copy.deepcopy(w)
		for i in range(len(Ut)):
			gradient[i] = self.mc_prox_dgd_one_gradient(Ut[i],d[i],w[i],lamb,eta,B)
		return gradient

	def gradient_new(self,Ut,w_now,d):
		w = copy.deepcopy(w_now)
		gradient = np.zeros_like(w)
		for i in range(len(Ut)):
			gradient[i] = self.one_gradient(Ut[i],d[i],w[i])
		return gradient
	
	def gradient_soft(self,Ut,w_now,d,lamb,eta,rho,m):
		w = copy.deepcopy(w_now)
		w_soft = copy.deepcopy(w_now)
		gradient = copy.deepcopy(w)
		for i in range(len(Ut)):
			ui = np.reshape(Ut[i],(len(Ut[i]),1))
			uti = ui.T
			wi = np.reshape(w[i],(len(w[i]),1))
			# gradient[i] = (ui@(uti@wi-d[i])-(rho)*wi+rho*(np.reshape(self.one_extra_L1(Ut[i],d[i],w_soft[i],lamb,1/rho),(len(wi),1)))).T[0]
			gradient[i] = (ui@(uti@wi-d[i])-(rho)*ui@uti@wi+rho*(np.reshape(self.one_extra_L1(Ut[i],d[i],w_soft[i],lamb,1/rho),(len(wi),1)))).T[0]
		return gradient
	
	def gradient_partial(self,Ut,w_now,d,lamb,eta,rho,m):
		w = copy.deepcopy(w_now)
		w_soft = copy.deepcopy(w_now)
		gradient = copy.deepcopy(w)
		for i in range(len(Ut)):
			ui = np.reshape(Ut[i],(len(Ut[i]),1))
			uti = ui.T
			ui_norm = ui/(np.linalg.norm(ui))
			uti_norm = ui_norm.T
			wi = np.reshape(w[i],(len(w[i]),1))
			P = ui_norm@uti_norm
			# P = (ui@uti)/((uti@ui)[0][0])
			# gradient[i] = (ui@(uti@wi-d[i])-(rho)*wi+rho*(np.reshape(self.one_extra_L1(Ut[i],d[i],w_soft[i],lamb,1/rho),(len(wi),1)))).T[0]
			gradient[i] = (ui@(uti@wi-d[i])-(rho)*P@wi + rho*(np.reshape(self.one_extra_L1(Ut[i],d[i],P@w_soft[i],lamb,1/rho),(len(wi),1)))).T[0]
		return gradient
	
	def gradient_partial_norm(self,Ut,w_now,d,lamb,eta,rho,m):
		gradient = self.gradient_partial(Ut,w_now,d,lamb,eta,rho,m)
		gradient_norm = copy.deepcopy(w_now)
		for i in range(len(gradient)):
			gradient_norm[i] = gradient[i]/np.linalg.norm(gradient[i])
		return gradient_norm
	
	def gradient_partial_each_param(self,Ut,w_now,d,lamb,eta,rho,m):
		w = copy.deepcopy(w_now)
		w_soft = copy.deepcopy(w_now)
		gradient = copy.deepcopy(w)
		for i in range(len(Ut)):
			ui = np.reshape(Ut[i],(len(Ut[i]),1))
			uti = ui.T
			ui_norm = ui/np.linalg.norm(ui)
			uti_norm = ui_norm.T
			wi = np.reshape(w[i],(len(w[i]),1))
			P = ui_norm@uti_norm
			# P = (ui@uti)/((uti@ui)[0][0])
			# gradient[i] = (ui@(uti@wi-d[i])-(rho)*wi+rho*(np.reshape(self.one_extra_L1(Ut[i],d[i],w_soft[i],lamb,1/rho),(len(wi),1)))).T[0]
			gradient[i] = (ui@(uti@wi-d[i])-(rho[i])*P@wi + rho[i]*(np.reshape(self.one_extra_L1(Ut[i],d[i],P@w_soft[i],lamb[i],1/rho[i]),(len(wi),1)))).T[0]
		return gradient
	
	def gradient_partial_norm_each_param(self,Ut,w_now,d,lamb,eta,rho,m):
		gradient = self.gradient_partial_each_param(Ut,w_now,d,lamb,eta,rho,m)
		gradient_norm = copy.deepcopy(w_now)
		for i in range(len(gradient)):
			gradient_norm[i] = gradient[i]/np.linalg.norm(gradient[i])
		return gradient_norm
	
	def gradient_mc(self,Ut,w_now,d,lamb,eta,rho,m):
		w = copy.deepcopy(w_now)
		w_soft = copy.deepcopy(w_now)
		gradient = copy.deepcopy(w)
		for i in range(len(Ut)):
			ui = np.reshape(Ut[i],(len(Ut[i]),1))
			uti = ui.T
			wi = np.reshape(w[i],(len(w[i]),1))
			gradient[i] = (ui@(uti@wi-d[i])-(rho)*wi+rho*(np.reshape(self.one_extra_L1(Ut[i],d[i],w_soft[i],lamb,1/rho),(len(wi),1)))).T[0]
			# gradient[i] = (ui@(uti@wi-d[i])-(rho)*ui@uti@wi+rho*(np.reshape(self.one_extra_L1(Ut[i],d[i],w_soft[i],lamb,1/rho),(len(wi),1)))).T[0]
		return gradient

	def gradient_NEXT(self,Ut,w_all,w_now,d,lamb,eta,rho,m,tau):
		w_all_next = copy.deepcopy(w_all)
		w = copy.deepcopy(w_now)
		w_soft = copy.deepcopy(w_now)
		gradient = copy.deepcopy(w)
		for i in range(len(Ut)):
			ui = np.reshape(Ut[i],(len(Ut[i]),1))
			uti = ui.T
			wi = np.reshape(w[i],(len(w[i]),1))
			w_next_i = np.reshape(w_all_next[i],(len(w[i]),1))
			gradient[i] = (ui@(uti@wi-d[i])-(rho)*wi+rho*(np.reshape(self.one_extra_L1(Ut[i],d[i],w_soft[i],lamb,1/rho),(len(wi),1)))).T[0] +tau*(wi-w_next_i).T[0]
			# gradient[i] = (ui@(uti@wi-d[i])-(rho)*ui@uti@wi+rho*(np.reshape(self.one_extra_L1(Ut[i],d[i],w_soft[i],lamb,1/rho),(len(wi),1)))).T[0]
		return gradient
		# for i in range(len(Ut)):
		# 	gradient[i] = self.one_gradient(Ut[i],d[i],w[i])-(rho)*w[i]+rho*(self.one_extra_L1(Ut[i],d[i],w[i],lamb,1/rho)) + tau*(w[i] - w_all_next[i])
		# return gradient

	def gradient_soft_nonconvex(self,Ut,w_now,d,lamb,eta,rho):
		w = copy.deepcopy(w_now)
		gradient = copy.deepcopy(w)
		for i in range(len(Ut)):
			gradient[i] = self.one_gradient(Ut[i],d[i],w[i])-(rho)*w[i]+rho*(self.one_extra_L1(Ut[i],d[i],w[i],lamb,1/rho))
		return gradient
	
	
	def gradient_L2_soft(self,Ut,w_now,d,lamb,eta,rho):
		w = copy.deepcopy(w_now)
		gradient = copy.deepcopy(w)
		for i in range(len(Ut)):
			gradient[i] = self.one_gradient(Ut[i],d[i],w[i])+rho*(self.one_extra_L1(Ut[i],d[i],w[i],lamb,1/rho))
		return gradient


	def one_gradient(self,U,d,w):
		ui = np.reshape(U,(len(U),1))
		uti = ui.T
		wi = np.reshape(w,(len(w),1))
		return (ui@(uti@wi-d)).T[0]
	
	def one_negative_gradient(self,U,d,w):
		return (d-U@w.T)*U
	
	def mc_prox_dgd_one_gradient(self,U,d,w,lamb,eta,B):
		return (U@w.T-d)*U -(lamb*(B**2))*w + (lamb*(B**2))*self.one_extra_L1(U,d,w,1/B,1/B)

	def one_extra_L1(self,Ut,d,w_now,lamb,eta):
		w = copy.deepcopy(w_now)
		for j in range(len(w)):
			if w[j] > 0 and eta*lamb < abs(w[j]):
				w[j] -= eta*lamb
			elif w[j] < 0 and eta*lamb < abs(w[j]):
				w[j] += eta*lamb
			else:
				w[j] = 0
		return w
	
	def all_extra_L1(self,Ut,d,w_all_now,lamb,eta):
		w_all = copy.deepcopy(w_all_now)
		w_next = copy.deepcopy(w_all_now)
		for i in range(len(Ut)):
			w_next[i] = self.one_extra_L1(Ut[i],d[i],w_all[i],lamb,eta)
		return w_next
	
	def all_extra_L1_each_param(self,Ut,d,w_all_now,lamb,eta):
		w_all = copy.deepcopy(w_all_now)
		w_next = copy.deepcopy(w_all_now)
		for i in range(len(Ut)):
			w_next[i] = self.one_extra_L1(Ut[i],d[i],w_all[i],lamb[i],eta)
		return w_next
	
	def all_extra_L1_each_param(self,Ut,d,w_all_now,lamb,eta):
		w_all = copy.deepcopy(w_all_now)
		w_next = copy.deepcopy(w_all_now)
		for i in range(len(Ut)):
			w_next[i] = self.one_extra_L1(Ut[i],d[i],w_all[i],lamb[i],eta)
		return w_next

	def all_extra_step_size_L1(self,Ut,d,w_all_now,lamb,eta_mat):
		w_all = copy.deepcopy(w_all_now)
		w_next = copy.deepcopy(w_all_now)
		for i in range(len(Ut)):
			w_next[i] = self.one_extra_L1(Ut[i],d[i],w_all[i],lamb,eta_mat[i][i])
		return w_next

	def l1_projection(self,Ut,d,w_all_now,K,eta):
		w_all = copy.deepcopy(w_all_now)
		w_next = copy.deepcopy(w_all_now)
		for i in range(len(Ut)):
			w_next[i] = self.euclidean_proj_l1ball(w_all[i],K)
		return w_next
	
	def euclidean_proj_simplex(self,v, s=1):
		assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
		n, = v.shape 
		if v.sum() == s and np.alltrue(v >= 0):
			return v
		u = np.sort(v)[::-1]
		cssv = np.cumsum(u)
		rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
		theta = (cssv[rho] - s) / (rho + 1.0)
		w = (v - theta).clip(min=0)
		return w


	def euclidean_proj_l1ball(self,v, s=1):
		assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
		n, = v.shape  # will raise ValueError if v is not 1-D
		u = np.abs(v)
		if u.sum() <= s:
			return v
		w = self.euclidean_proj_simplex(u, s=s)
		w *= np.sign(v)
		return w

	def all_extra_L1_atc(self,Ut,d,w_all_now,lamb,eta,c):
		w_all = copy.deepcopy(w_all_now)
		w_next = copy.deepcopy(w_all_now)
		for i in range(len(Ut)):
			w_next[i] = self.one_extra_L1(Ut[i],d[i],w_all[i],lamb,eta)
		return c@w_next

	def one_extra_mc(self,Ut,d,w,lamb,eta,rho):
		w_next = copy.deepcopy(w)
		for j in range(len(w)):
			if abs(w[j]) <= eta*lamb:
				w_next[j] = 0
			elif eta*lamb < abs(w[j]) and abs(w[j]) < lamb/rho:
				w_next[j] = w[j]*(abs(w[j])-eta*lamb)/(abs(w[j])*(1-eta*rho))
			elif lamb/rho <= abs(w[j]):
				w_next[j] = w[j]
			else:
				print("banana")
		return w_next
	
	def one_extra_scad(self,Ut,d,w,lamb,eta,rho):
		w_next = copy.deepcopy(w)
		for j in range(len(w)):
			if abs(w[j]) <= 2*lamb:
				w_next[j] = (w[j]/abs(w[j]))*max([abs(w[j])-lamb,0])
			elif 2*lamb < abs(w[j]) and abs(w[j]) <= lamb*rho:
				w_next[j] = ((rho-1)*w[j]-(w[j]/(abs(w[j])))*lamb*rho)/(rho-2)
			elif lamb*rho < abs(w[j]):
				w_next[j] = w[j]
			else:
				print("banana")
		return w_next
	
	def all_extra_mc(self,Ut,d,w_all_1,lamb,eta,rho):
		w_all = copy.deepcopy(w_all_1)
		w_next = copy.deepcopy(w_all_1)
		for i in range(len(Ut)):
			w_next[i] = self.one_extra_mc(Ut[i],d[i],w_all[i],lamb,eta,rho)
		return w_next

	def all_extra_scad(self,Ut,d,w_all_1,lamb,eta,rho):
		w_all = copy.deepcopy(w_all_1)
		w_next = copy.deepcopy(w_all_1)
		for i in range(len(Ut)):
			w_next[i] = self.one_extra_scad(Ut[i],d[i],w_all[i],lamb,eta,rho)
		return w_next
	
	def all_extra_mc_convex(self,Ut,d,w_all_1,lamb,eta,rho):
		w_all = copy.deepcopy(w_all_1)
		w_next = copy.deepcopy(w_all_1)
		for i in range(len(Ut)):
			ui = Ut[i]/np.linalg.norm(Ut[i], ord=2)
			w_next[i] = w_next[i] - ui@(w_next[i] - self.one_extra_mc(Ut[i],d[i],w_all[i],lamb,eta,rho))*ui
		return w_next
	
	def all_extra_mc_convex_2(self,Ut,d,w_all_1,lamb,eta,rho):
		#not complete
		w_all = copy.deepcopy(w_all_1)
		w_next = copy.deepcopy(w_all_1)
		for i in range(len(Ut)):
			ui = Ut[i]/np.linalg.norm(Ut[i], ord=2)
			w_next[i] = w_next[i] - ui@(w_next[i] - self.one_extra_mc(Ut[i],d[i],w_all[i],lamb,eta,rho))*ui
		return w_next