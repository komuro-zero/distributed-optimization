#functions used in each simulations
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt

class functions():
	def w_star(self,N,sparse_percentage):
		w_star = randn(N,1)     #０に近い値が出力される。つまり平均は0
		before_length = len(w_star)
		after_length = before_length
		while (after_length / before_length) > sparse_percentage:
			a = randint(0,N-1)  #randintは整数で返す
			w_star[a] = 0
			after_length = len(np.unique(w_star))-1  #重複要素の削除
		return w_star

	def db(self,x, y):
		db = 10 * np.log10(x / y)    
		return db[0][0]

	def centralized_gradient_descent(self,Ut,d,w,w_star,L2,eta,iteration):
		error = []
		times = []
		one_error =0
		U = Ut.T
		for i in range(iteration):
			w = w - 2*eta*(np.dot(U,(np.dot(Ut,w)-d)))
			one_error = np.dot((w-w_star).T,w-w_star)[0]
			error.append(self.db(one_error,L2))
			times.append(i)
			one_error = 0
		#plt.plot(times,error,label = "centralized gradient descent")
		return error

	def centralized_L1(self,Ut,d,w,w_star,L2,lamb,eta,iteration):
		error = []
		times = []
		one_error =0
		U = Ut.T
		for i in range(iteration):
			w = w - 2*eta*(np.dot(U,(np.dot(Ut,w)-d)))
			for j in range(len(w)):
				if w[j] > 0 and lamb < abs(w[j]):
					w[j] -= lamb
				elif w[j] < 0 and lamb < abs(w[j]):
					w[j] += lamb
				else:
					w[j] = 0
			one_error = np.dot((w-w_star).T,w-w_star)[0]
			error.append(self.db(one_error,L2))
			times.append(i)
			one_error = 0
		#exec("plt.plot(times,error,label = 'centralized_L1')")
		return error

	def centralized_mc(self,Ut,d,w,w_star,L2,lamb,eta,rho,iteration):
		error = []
		times = []
		one_error =0
		U = Ut.T
		for i in range(iteration):
			w = w - 2*eta*(np.dot(U,(np.dot(Ut,w)-d)))
			for j in range(len(w)):
				if abs(w[j]) < eta*lamb:
					w[j] = 0
				elif eta*lamb < abs(w[j]) and abs(w[j]) < lamb/rho:
					w[j] = w[j]*(abs(w[j])-eta*lamb)/(abs(w[j])*(1-eta*rho))
				elif lamb/rho <= abs(w[j]):
					w[j] = w[j]
				else:
					print("banana")
			one_error = np.dot((w-w_star).T,w-w_star)[0]
			error.append(self.db(one_error,L2))
			times.append(i)
			one_error = 0
		#plt.plot(times,error,label = "centralized_mc")
		return error

	def one_gradient_descent(self,Ut,d,w,eta):
		U = Ut.T
		w = w - 2*eta*((U*(np.dot(Ut,w)-d)))
		return w

	def one_L1(self,Ut,d,w,lamb,eta):
		U = Ut.T
		w = w - 2*eta*((U*(np.dot(Ut,w)-d)))
		for j in range(len(w)):
			if w[j][0] > 0 and lamb < abs(w[j][0]):
				w[j][0] -= lamb
			elif w[j][0] < 0 and lamb < abs(w[j][0]):
				w[j][0] += lamb
			else:
				w[j][0] = 0
		return w

	def one_mc(self,Ut,d,w,lamb,eta,rho):
		U = Ut.T
		w = w - 2*eta*((U*(np.dot(Ut,w)-d)))
		for j in range(len(w)):
			if abs(w[j][0]) < eta*lamb:
				w[j][0] = 0
			elif eta*lamb < abs(w[j][0]) and abs(w[j][0]) < lamb/rho:
				w[j][0] = w[j][0]*(abs(w[j][0])-eta*lamb)/(abs(w[j][0])*(1-eta*rho))
			elif lamb/rho <= abs(w[j][0]):
				w[j][0] = w[j][0]
			else:
				print("banana")
		return w

	def distributed_gradient_descent(self,Ut,d,w_star,L2,N,m,r_i,eta,iteration):
		for i in range(1,m+1,1):
			exec("d_%d = np.dot(np.reshape(Ut[%d],(1,N)),w_star)" % (i,i-1))
			exec("w_%d = randn(N,1) " % (i))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
		average_error = []
		times = []
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_gradient_descent(np.reshape(Ut[%d],(1,N)),d_%d,w_%d,eta)" % (j,j-1,j,j))
				#exec("print(w_%d)" % (j))
				#print("=================")
			for k in range(1,m+1,1):
				exec("w_next_%d = w_%d" % (k,k))
				for l in range(1,r_i+1	,1):
					if k+l <= m:
						exec("w_next_%d += w_%d" % (k,k+l))
					else:
						exec("w_next_%d += w_%d" % (k,k+l-m))
				exec("w_next_%d = w_next_%d/%d" % (k,k,1+r_i))
				exec("w_%d = w_next_%d" % (k,k))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star).T,w_%d-w_star)[0]" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i,j,i))
			average_error[i] = average_error[i]/m
		exec("plt.plot(times,average_error,label = 'distributed gradient descent')")
		return average_error

	def distributed_L1(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,iteration):	
		for i in range(1,m+1,1):
			exec("d_%d = np.dot(np.reshape(Ut[%d],(1,N)),w_star)" % (i,i-1))
			exec("w_%d = randn(N,1) " % (i))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
		average_error = []
		times = []
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_L1(np.reshape(Ut[%d],(1,N)),d_%d,w_%d,lamb,eta)" % (j,j-1,j,j))
			for k in range(1,m+1,1):
				for l in range(1,r_i+1,1):
					if k+l <= m:
						exec("w_next_%d += w_%d" % (k,k+l))
					else:
						exec("w_next_%d += w_%d" % (k,k+l-m))
				exec("w_next_%d = w_next_%d/%d" % (k,k,1+r_i))
				exec("w_%d = w_next_%d" % (k,k))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star).T,w_%d-w_star)[0]" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i,j,i))
			average_error[i] = average_error[i]/m
		exec("plt.plot(times,error,label = 'distributed L1')")
		return average_error


	def distributed_mc(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration):
		for i in range(1,m+1,1):
			exec("d_%d = np.dot(np.reshape(Ut[%d],(1,N)),w_star)" % (i,i-1))
			exec("w_%d = randn(N,1) " % (i))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
		average_error = []
		times = []
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_mc(np.reshape(Ut[%d],(1,N)),d_%d,w_%d,lamb,eta,rho)" % (j,j-1,j,j))
			for k in range(1,m+1,1):
				for l in range(1,r_i+1,1):
					if k+l <= m:
						exec("w_next_%d += w_%d" % (k,k+l))
					else:
						exec("w_next_%d += w_%d" % (k,k+l-m))
				exec("w_next_%d = w_next_%d/%d" % (k,k,1+r_i))
				exec("w_%d = w_next_%d" % (k,k))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star).T,w_%d-w_star)[0]" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i,j,i))
			average_error[i] = average_error[i]/m
		#exec("plt.plot(times,average_error,label = 'distributed mc average')")
		return average_error
	
	def distributed_gradient_descent_wj(self,Ut,d,w_star,L2,N,m,r_i,eta,iteration):
		for i in range(1,m+1,1):
			exec("d_%d = np.dot(np.reshape(Ut[%d],(1,N)),w_star)" % (i,i-1))
			exec("w_%d = randn(N,1) " % (i))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
		average_error = []
		times = []
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_gradient_descent(np.reshape(Ut[%d],(1,N)),d_%d,w_%d,eta)" % (j,j-1,j,j))
				#exec("print(w_%d)" % (j))
				#print("=================")
			for k in range(1,m+1,1):
				exec("w_next_%d = w_%d" % (k,k))
				for l in range(1,r_i+1	,1):
					if k+l <= m:
						exec("w_next_%d += w_%d" % (k,k+l))
					else:
						exec("w_next_%d += w_%d" % (k,k+l-m))
				exec("w_next_%d = w_next_%d/%d" % (k,k,1+r_i))
				exec("w_%d = w_next_%d" % (k,k))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star).T,w_%d-w_star)[0]" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i,j,i))
			average_error[i] = average_error[i]/m
		exec("plt.plot(times,average_error,label = 'distributed gradient descent')")
		return average_error

	def distributed_L1(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,iteration):	
		for i in range(1,m+1,1):
			exec("d_%d = np.dot(np.reshape(Ut[%d],(1,N)),w_star)" % (i,i-1))
			exec("w_%d = randn(N,1) " % (i))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
		average_error = []
		times = []
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_L1(np.reshape(Ut[%d],(1,N)),d_%d,w_%d,lamb,eta)" % (j,j-1,j,j))
			for k in range(1,m+1,1):
				for l in range(1,r_i+1,1):
					if k+l <= m:
						exec("w_next_%d += w_%d" % (k,k+l))
					else:
						exec("w_next_%d += w_%d" % (k,k+l-m))
				exec("w_next_%d = w_next_%d/%d" % (k,k,1+r_i))
				exec("w_%d = w_next_%d" % (k,k))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star).T,w_%d-w_star)[0]" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i,j,i))
			average_error[i] = average_error[i]/m
		exec("plt.plot(times,error,label = 'distributed L1')")
		return average_error


	def distributed_mc(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration):
		for i in range(1,m+1,1):
			exec("d_%d = np.dot(np.reshape(Ut[%d],(1,N)),w_star)" % (i,i-1))
			exec("w_%d = randn(N,1) " % (i))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
		average_error = []
		times = []
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_mc(np.reshape(Ut[%d],(1,N)),d_%d,w_%d,lamb,eta,rho)" % (j,j-1,j,j))
			for k in range(1,m+1,1):
				for l in range(1,r_i+1,1):
					if k+l <= m:
						exec("w_next_%d += w_%d" % (k,k+l))
					else:
						exec("w_next_%d += w_%d" % (k,k+l-m))
				exec("w_next_%d = w_next_%d/%d" % (k,k,1+r_i))
				exec("w_%d = w_next_%d" % (k,k))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star).T,w_%d-w_star)[0]" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i,j,i))
			average_error[i] = average_error[i]/m
		#exec("plt.plot(times,average_error,label = 'distributed mc average')")
		return average_error