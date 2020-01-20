import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
import copy
from modules.base_functions import base
import numpy.linalg as LA
import pywt


class update_functions(base):
	def centralized_gradient_descent(self,Ut,d,w,w_star,L2,eta,iteration):
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

	def centralized_L1(self,Ut,d,w,w_star,L2,lamb,eta,iteration):
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		U = Ut.T
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
	
	def centralized_mc_twin(self,Ut,d,w,w_star,L2,lamb,eta,rho,iteration,m):
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		prox = np.zeros_like(w)
		U = Ut.T
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
		exec("plt.plot(times,error,label = 'centralized mc twin prox nonconvex')")
		return error,w

	def centralized_mc(self,Ut,d,w,w_star,L2,lamb,eta,rho,iteration):
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)] 
		times = [0]
		one_error =0
		U = Ut.T
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
		plt.plot(times,error,label = "centralized_mc")
		return error,w

	def one_gradient_descent(self,Ut,d,w,eta):
		U = Ut.T
		w = w - eta*((U*(np.dot(Ut,w)-d)))
		return w

	def all_gradient_descent(self,Ut,w_next,d,w_all,eta):
		return 

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
		w_all_next = copy.deepcopy(w_all)
		w_all_iter = copy.deepcopy(w_all)
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_iter,w_star,N,L2,m))
			w_all_next = self.all_gradient_descent(Ut,w_all_next,d,w_all_iter,eta)
			w_all_iter = (1/(r_i+1))*(c@w_all_next)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'new distributed gradient descent')
		#return average_error,w_all

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
		w_all_next = copy.deepcopy(w_all)
		w_all_before = copy.deepcopy(w_all)
		w_all_prox = copy.deepcopy(w_all)
		c = (1/(r_i+1))*c
		c_tilde = (1/2)*(np.eye(len(c))+c)

		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_prox = self.extra_share(Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i)
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_mc(Ut,w_all_next,d,w_all_prox,lamb,eta,rho)
			if i % 1000 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'PG-EXTRA with MC penalty')
		return np.mean(w_all_next,axis = 0)

	def pg_extra_mc_soft(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
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
				w_all_prox = c@w_all_before-eta*(self.gradient_soft(Ut,w_all_before,d,lamb,eta,rho,m))
			else:
				w_all_prox = c@w_all_next + w_all_prox_before - c_tilde@w_all_before - eta*(self.gradient_soft(Ut,w_all_next,d,lamb,eta,rho,m)-self.gradient_soft(Ut,w_all_before,d,lamb,eta,rho,m))
			w_all_before = copy.deepcopy(w_all_next)
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,eta,lamb)
			if i % 100 == 0:
				print(f"iteration: {i}")
		times = range(len(average_error))
		# plt.plot(times,average_convergence)
		# plt.title("convergence over iteration")
		# plt.show()
		plt.plot(times,average_error,label = 'PG-EXTRA with MC penalty')
		return average_error

	def pg_extra_mc_soft_nonconvex(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
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
		plt.plot(times,average_error,label = 'extra mc nonconvex')
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
		for i in range(iteration):
			average_error.append(self.error_distributed(w_all_before,w_star,N,L2,m))
			w_all_between = copy.deepcopy(w_all_next)
			w_all_next = self.extra_share(Ut,d,c,c_tilde,w_all_next,w_all_before,eta,i)
			w_all_before = copy.deepcopy(w_all_between)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'EXTRA')
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
	
	def extra_l1_share(self,Ut,d,c,c_tilde,w_all_next,w_all_before,w_all_prox,eta,i):
		if i == 0:
			return c@w_all_next - eta*(self.gradient(Ut,w_all_next,d))
		else:
			return c@w_all_next + w_all_prox - c_tilde@w_all_before - eta*(self.gradient(Ut,w_all_next,d)-self.gradient(Ut,w_all_before,d))
	
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

	def pg_extra_l1(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all):
		average_error = []
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
			w_all_next = self.all_extra_L1(Ut,d,w_all_prox,lamb,eta)
			if i %100 == 0:
				print("iteration:",i)
		times = range(len(average_error))
		plt.plot(times,average_error,label = 'PG-EXTRA with L1 penalty')
		return average_error

	def gradient(self,Ut,w_now,d):
		w = copy.deepcopy(w_now)
		gradient = copy.deepcopy(w)
		for i in range(len(Ut)):
			gradient[i] = self.one_gradient(Ut[i],d[i],w[i])
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
			gradient[i] = (Ut[i]@w[i].T-d[i])*Ut[i]-((rho)*Ut[i]@w[i].T)*Ut[i]+rho*(self.one_extra_L1(Ut[i],d[i],w_soft[i],lamb,1/rho))
		return gradient

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
		return (U@w.T-d)*U

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
	
	def all_extra_mc(self,Ut,w_next,d,w_all,lamb,eta,rho):
		for i in range(len(Ut)):
			w_next[i] = self.one_extra_mc(Ut[i],d[i],w_all[i],lamb,eta,rho)
		return w_next