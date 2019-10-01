#functions used in each simulations
import numpy as np
from numpy.random import *
import matplotlib.pyplot as plt
<<<<<<< HEAD
=======
import numpy.linalg as LA
>>>>>>> bd7fd769a1864b8fc9cade4e0229cedb02533b04

#np.random.seed(0)

class functions():
	def w_star(self,N,sparse_percentage):
		w_star = randn(N,1)     #平均は0
		before_length = len(w_star)
		after_length = before_length
		while (after_length / before_length) > sparse_percentage:
			a = randint(0,N-1)  
			w_star[a] = 0
			after_length = len(np.unique(w_star))-1  #重複要素の削除
		return w_star
	
	def w_star_weakly_sparse(self,N,sparse_percentage,how_weak):
		w_star = randn(N,1)     #平均は0
		before_length = len(w_star)
		after_length = before_length
		while (after_length / before_length) > sparse_percentage:
			a = randint(0,N-1)  
			w_star[a] = 0
			after_length = len(np.unique(w_star))-1  #重複要素の削除
		w_star_prox = np.zeros((len(w_star),1))
		for i in range(len(w_star)):
			if w_star[i] == 0:
				w_star_prox[i] = randn()
		w_star_prox = w_star_prox/np.dot(w_star_prox.T,w_star_prox)
		w_star = w_star/np.dot(w_star.T,w_star)
		w_star += how_weak*w_star_prox
		return w_star
	
	def w_star_weakly_sparse_sample(self,N,sparse_percentage):
		w_star = self.w_star(N,sparse_percentage)
		for i in range(len(w_star)):
			if w_star[i] == 0:
				w_star[i] = 0.0001
			else:
				w_star[i] = 1
		return w_star

	def db(self,x, y):
		db = 10 * np.log10(x / y)    
		return db[0][0]

	def rho_checker(self,rho,lamb,eta):
		if lamb/rho <= eta*lamb:
			print("faulty rho")
			exit()
<<<<<<< HEAD
=======
	
	def U_eigenvalue(self,U_all):
		square = U_all.T@U_all
		eigenvalue = LA.eig(square)[0]
		smallest_eigen = min(eigenvalue)
		biggest_eigen = max(eigenvalue)
		return smallest_eigen,biggest_eigen
	
	def params_checker(self,rho,lamb,eta,U_all,B,m,N):
		small_eig, big_eig = self.U_eigenvalue(U_all)
		if rho > small_eig:
			print("rho is bigger than the smallest eigen")
			exit()
		elif eta < 2/big_eig:
			print("eta may be too big")
		self.rho_checker(rho,lamb,eta)
		self.lipschitz_checker(U_all,B,m,eta)
		self.distributed_lipschitz_checker(U_all,B,m,eta,N)
	
	def lipschitz_checker(self,U,B,m,eta):
		X = U@U.T-(B)*np.eye(m)
		U, s, V = np.linalg.svd(X)
		lpz = max(s)
		if lpz/2 < eta:
			print(f"eta must be smaller than {lpz/2}")
			exit()
	
	def distributed_lipschitz_checker(self,U,B,m,eta,N):
		for u in U:
			ut = np.reshape(u,(len(u),1)) 
			um = np.reshape(u,(1,len(u)))
			x = ut@um-(B**2)*np.eye(N)
			U, s, V = np.linalg.svd(x)
			lpz = max(LA.eig(x)[0])
			lpz = max(s)/2
			print(lpz)
			if lpz < eta:
				print(f"eta must be smaller than {lpz/2}")
				exit()

>>>>>>> bd7fd769a1864b8fc9cade4e0229cedb02533b04

	def directed_graph(self,m,r_i):
		c = np.eye(m) 
		for j in range(m):
			counter = 0
			while counter != r_i:
				random_num = np.random.randint(0,m)
				if c[j][random_num] != 1:
					c[j][random_num] = 1
					counter += 1
		return c
	
	def disjointed_directed_graph(self,m,r_i):
		a = [1]*r_i
		b = [0]*(m-r_i)
		a.extend(b)
		c = [1]*r_i
		d = [0]*(m-r_i)
		d.extend(c)
		aa = [a]*int(m/2)
		dd = [d]*int(m/2)
		dd.extend(aa)
		result = np.array(dd)
		for i in range(len(result)):
			result[i][i]=1
		return result
	
	def make_range(self,m):
		result =[]
		for i in range(m):
			result.append(i)
		return result

	def make_checklist(self,r_i,m):
		untidy_result = self.make_range(m)
		for i in range(r_i-1):
			untidy_result[0:0] = self.make_range(m)
		return sorted(untidy_result)

	def horizontal_checker(self,c,place,r_i):
		count = 0
		for row in c:
			if row[place] == 1:
				count += 1
		if count == r_i+1:
			result = False
		elif count < r_i+1:
			result = True
		else:
			print(c)
			print("faulty horizontal number")
			exit()
		return result

	def undirected_graph(self,m,r_i):
		c = np.eye(m)
		checklist = self.make_checklist(r_i,m)
		for i in range(m):
			connected = sum(x == 1 for x in c[i])
			while connected < r_i:
				select = np.random.randint(len(checklist))
				connect_node = checklist[select]
				if c[i][connect_node] != 1 and self.horizontal_checker(c,connect_node,r_i):
					c[i][connect_node] = 1
					connected += 1
					c[connect_node][i] =1
					checklist.pop(select)
		return c

	def find_ones(self,row):
		result = [i for i, element in enumerate(row) if element == 1]
		return result

	def all_connections(self,c_list):
		result = []
		for row in c_list:
			result.append(self.find_ones(row))
		return result

	def delete_self(self,all_connection):
		for i in range(len(all_connection)):
			all_connection[i].remove(i)
		return all_connection

	def disjoint_checker(self,c,m):
		c_list = c.tolist()
		all_connection = self.all_connections(c_list)
		adjacent_matrix = self.delete_self(all_connection)
		check_node = adjacent_matrix[0]
		connected_node = adjacent_matrix[0]
		last_length = len(connected_node)
		check_node_box = []
		count = 1

		flag = True
		while flag:
			for i in check_node:
				check_node_box[0:0] = (all_connection[i])
			common_part = list(set(check_node_box) & set(connected_node))
			check_node = list(set(check_node_box) - set(common_part))
			connected_node[0:0] = check_node_box
			connected_node = list(set(connected_node))
			check_node_box = []
			now_length = len(connected_node)
			if now_length == last_length:
				flag = False
			count += 1
			if count > m or len(connected_node) == m:
				flag = False
			last_length = now_length
		if len(connected_node) == m:
			print("Your graph is not disjoint")
		else:
			print("Your graph is disjoint. connected nodes:",now_length)

	def w_average(self,w_all,w_star):
		result = [0]*len(w_all[0])
		for row in w_all:
			result += row
		result = result/len(w_all)
		x = range(len(result))
<<<<<<< HEAD
		#fig = plt.figure()
		#plt.plot(x,w_star,color = "black",label = "w_star")
		#plt.plot(x,result,color = "red",label = "w_average")
		#plt.show()
=======
		fig = plt.figure()
		plt.plot(x,w_star,color = "black",label = "w_star")
		plt.plot(x,result,color = "red",label = "w_average")
		plt.show()
>>>>>>> bd7fd769a1864b8fc9cade4e0229cedb02533b04
		return result
	
	def make_w(self,m,N):
		w = randn(N,1).T
		for i in range(m-1):
			w = np.concatenate((w,randn(N,1).T))
		return w

<<<<<<< HEAD
=======
	def make_variables(self,N,m,sparsity_percentage,how_weakly_sparse,w_noise):
		w = randn(N,1)
		w_star = self.w_star_weakly_sparse(N,sparsity_percentage,how_weakly_sparse)
		#w_star = w_star(N,sparsity_percentage)
		U_all = randn(m,N)
		w_star_noise = w_star +randn(N,1)*(10**-(w_noise/10))
		#w_star_noise = w_star_noise/np.dot(w_star_noise.T,w_star_noise)
		d_all = np.dot(U_all,w_star_noise)
		L2 = np.dot(w_star.T,w_star)

		return w,w_star,U_all,d_all,L2

>>>>>>> bd7fd769a1864b8fc9cade4e0229cedb02533b04
	def centralized_gradient_descent(self,Ut,d,w,w_star,L2,eta,iteration):
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
		one_error =0
		U = Ut.T
		for i in range(iteration):
			w = w - 2*eta*(np.dot(U,(np.dot(Ut,w)-d)))
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
			times.append(i+1)
		exec("plt.plot(times,error,label = 'centralized_L1')")
		return error,w

	def centralized_mc(self,Ut,d,w,w_star,L2,lamb,eta,rho,iteration):
		error = [self.db(np.dot((w-w_star).T,w-w_star)[0],L2)]
		times = [0]
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
<<<<<<< HEAD
					print("banana")
=======
					print("eta*lamb",eta*lamb,"lamb/rho =",lamb/rho,"w = ",w[j])
>>>>>>> bd7fd769a1864b8fc9cade4e0229cedb02533b04
			one_error = np.dot((w-w_star).T,w-w_star)[0]
			error.append(self.db(one_error,L2))
			times.append(i+1)
			one_error = 0
		plt.plot(times,error,label = "centralized_mc")
		return error,w

	def one_gradient_descent(self,Ut,d,w,eta):
		U = Ut.T
		w = w - 2*eta*((U*(np.dot(Ut,w)-d)))
		return w

	def one_L1(self,Ut,d,w,lamb,eta):
		U = Ut.T
		w = w - 2*eta*((U*(np.dot(Ut,w)-d)))
		for j in range(len(w)):
			if w[j] > 0 and lamb < abs(w[j]):
				w[j] -= lamb
			elif w[j] < 0 and lamb < abs(w[j]):
				w[j] += lamb
			else:
				w[j] = 0
		return w

	def one_mc(self,Ut,d,w,lamb,eta,rho):
		U = Ut.T
		w = w - 2*eta*((U*(np.dot(Ut,w)-d)))
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

	def distributed_gradient_descent(self,Ut,d,w_star,L2,N,m,r_i,eta,iteration,c,w_all_f):
		average_error = [0]
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel()),L2)"%(i,i))
		average_error[0] = average_error[0]/m
		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_gradient_descent(u_%d,d_%d,w_%d,eta)" % (j,j,j,j))
				exec("w_next_%d = w_%d" % (j,j))
			exec("w_all = [w_next_1]")
			for j in range(2,m+1):
				exec("w_all = np.concatenate((w_all,[w_next_%d]))" %(j))
			exec("average = (1/(r_i+1))*np.dot(c,w_all)")
			for j in range(1,m+1,1):
				exec("w_%d = average[%d]" % (j,j-1))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel())" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i+1)
		exec("w_all_f = w_all")
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		plt.plot(times,average_error,label = 'distributed gradient descent')
		plt.legend()
<<<<<<< HEAD
		exec("w_average = self.w_average(w_all,w_star)")		
		return average_error,w_all_f

=======
		return average_error,w_all_f

	"""def distributed_gradient_descent2(self,Ut,d,w_star,L2,N,m,r_i,eta,iteration,c,w_all_f):
		average_error = [0]
		w = self.make_w(m,N)
		w_next = w
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel()),L2)"%(i,i))
		average_error[0] = average_error[0]/m
		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				w[j] = self.one_gradient_descent(Ut[j],d[j][0],np.reshape(w_all_f[j],(N,1)).ravel(),eta)
				w_next[j] = w[j]
			exec("average = (1/(r_i+1))*np.dot(c,w_next)")
			w = average
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel())" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i+1)
		exec("w_all_f = w_all")
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		plt.plot(times,average_error,label = 'distributed gradient descent')
		plt.legend()
		return average_error,w_all_f"""

>>>>>>> bd7fd769a1864b8fc9cade4e0229cedb02533b04
	def distributed_L1(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,iteration,c,w_all_f):	
		average_error = [0]
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel()),L2)"%(i,i))
		average_error[0] = average_error[0]/m

		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_L1(u_%d,d_%d,w_%d,lamb,eta)" % (j,j,j,j))
				exec("w_next_%d = w_%d" % (j,j))
			exec("w_all = [w_next_1]")
			for j in range(2,m+1):
				exec("w_all = np.concatenate((w_all,[w_next_%d]))" %(j))
			exec("average = (1/(r_i+1))*np.dot(c,w_all)")
			for j in range(1,m+1,1):
				exec("w_%d = average[%d]" % (j,j-1))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel())" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i+1)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		plt.plot(times,average_error,label = 'distributed L1')
		plt.legend()
<<<<<<< HEAD
		exec("w_average = self.w_average(w_all,w_star)")
=======
>>>>>>> bd7fd769a1864b8fc9cade4e0229cedb02533b04
		return average_error

	def distributed_mc(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all_f):
		average_error = [0]
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			#exec("error_w_%d = [self.db(np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel()),L2)]" % (i,i,i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel()),L2)"%(i,i))
		average_error[0] = average_error[0]/m
		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_mc(u_%d,d_%d,w_%d,lamb,eta,rho)" % (j,j,j,j))
				exec("w_next_%d = w_%d" % (j,j))
			exec("w_all = [w_next_1]")
			for j in range(2,m+1):
				exec("w_all = np.concatenate((w_all,[w_next_%d]))" %(j))
			exec("average = (1/(r_i+1))*np.dot(c,w_all)")
			for j in range(1,m+1,1):
				exec("w_%d = average[%d]" % (j,j-1))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-w_star.ravel()).T,w_%d-w_star.ravel())" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2))" % (j,j))
			times.append(i+1)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		#for i in range(1,m+1):
		#	exec("plt.plot(times,error_w_%d)"%(i))
		plt.plot(times,average_error,label = 'distributed mc')
		plt.legend()
<<<<<<< HEAD
		exec("w_average = self.w_average(w_all,w_star)")		
=======
>>>>>>> bd7fd769a1864b8fc9cade4e0229cedb02533b04
		return average_error

	def distributed_gradient_descent_wj(self,Ut,d,w_star,L2,N,m,r_i,eta,iteration,c,w_all_f,wj):
		average_error = [0]
		L2wj = np.dot(wj.T,wj)
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-wj.ravel()).T,w_%d-wj.ravel()),L2wj)"%(i,i))
		average_error[0] = average_error[0]/m
		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_gradient_descent(u_%d,d_%d,w_%d,eta)" % (j,j,j,j))
				exec("w_next_%d = w_%d" % (j,j))
			exec("w_all = [w_next_1]")
			for j in range(2,m+1):
				exec("w_all = np.concatenate((w_all,[w_next_%d]))" %(j))
			exec("average = (1/(r_i+1))*np.dot(c,w_all)")
			for j in range(1,m+1,1):
				exec("w_%d = average[%d]" % (j,j-1))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-wj.ravel()).T,w_%d-wj.ravel())" % (j,j,j))
				#exec("print(one_error_%d)"%(j))
				exec("error_w_%d.append(self.db(one_error_%d,L2wj))" % (j,j))
			times.append(i+1)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		exec("plt.plot(times,average_error,label = 'wj distributed gradient descent')")
<<<<<<< HEAD
		exec("w_average = self.w_average(w_all,wj)")		
=======
>>>>>>> bd7fd769a1864b8fc9cade4e0229cedb02533b04
		return average_error

	def distributed_L1_wj(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,iteration,c,w_all_f,wj):	
		average_error = [0]
		L2wj = np.dot(wj.T,wj)
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-wj.ravel()).T,w_%d-wj.ravel()),L2wj)"%(i,i))
		average_error[0] = average_error[0]/m

		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_L1(u_%d,d_%d,w_%d,lamb,eta)" % (j,j,j,j))
				exec("w_next_%d = w_%d" % (j,j))
			exec("w_all = [w_next_1]")
			for j in range(2,m+1):
				exec("w_all = np.concatenate((w_all,[w_next_%d]))" %(j))
			exec("average = (1/(r_i+1))*np.dot(c,w_all)")
			for j in range(1,m+1,1):
				exec("w_%d = average[%d]" % (j,j-1))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-wj.ravel()).T,w_%d-wj.ravel())" % (j,j,j))
				exec("error_w_%d.append(self.db(one_error_%d,L2wj))" % (j,j))
			times.append(i+1)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		exec("plt.plot(times,average_error,label = 'wj distributed L1')")
		return average_error

	def distributed_mc_wj(self,Ut,d,w_star,L2,N,m,r_i,lamb,eta,rho,iteration,c,w_all_f,wj):
		average_error = [0]
		L2wj = np.dot(wj.T,wj)
		for i in range(1,m+1,1):
			exec("u_%d = Ut[%d]" % (i,i-1))
			exec("d_%d = d[%d][0]" % (i,i-1))
			exec("w_%d = np.reshape(w_all_f[%d],(N,1)).ravel() " % (i,i-1))
			exec("w_next_%d = w_%d" % (i,i))
			exec("error_w_%d = []" % (i))
			exec("one_error_%d =[]" % (i))
			exec("average_error[0] += self.db(np.dot((w_%d-wj.ravel()).T,w_%d-wj.ravel()),L2wj)"%(i,i))
		average_error[0] = average_error[0]/m
		times = [0]
		for i in range(iteration):
			for j in range(1,m+1,1):
				exec("w_%d = self.one_mc(u_%d,d_%d,w_%d,lamb,eta,rho)" % (j,j,j,j))
				exec("w_next_%d = w_%d" % (j,j))
			exec("w_all = [w_next_1]")
			for j in range(2,m+1):
				exec("w_all = np.concatenate((w_all,[w_next_%d]))" %(j))
			exec("average = (1/(r_i+1))*np.dot(c,w_all)")
			for j in range(1,m+1,1):
				exec("w_%d = average[%d]" % (j,j-1))
			for j in range(1,m+1,1):
				exec("one_error_%d = np.dot((w_%d-wj.ravel()).T,w_%d-wj.ravel())" % (j,j,j))
				
				exec("error_w_%d.append(self.db(one_error_%d,L2wj))" % (j,j))
			times.append(i+1)
		for i in range(iteration):
			exec("average_error.append(error_w_1[%d])" % (i))
			for j in range(2,m+1,1):
				exec("average_error[%d] += error_w_%d[%d]" % (i+1,j,i))
			average_error[i+1] = average_error[i+1]/m
		exec("plt.plot(times,average_error,label = 'wj distributed mc')")
		return average_error
