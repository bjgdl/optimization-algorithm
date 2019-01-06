import numpy as np 

class Rank_One():

	def __init__(self,Q,X0):
		self.Q = Q
		self.X0 = X0
		self.H0 = np.eye(X0.size)

	def gk(self,Q,X):
		gk = np.dot(Q,X)
		return gk

	def dk(self,H,g):
		dk=-1*np.dot(H,g)
		return dk

	def alpha_k(self,g,d,Q):
		alpha = -1*(np.dot(g.transpose(),d))/np.dot(np.dot(d.transpose(),Q),d)
		return alpha


	def func(self):
		x = self.X0
		H = self.H0
		while(1):
			g = self.gk(self.Q,x)
			if np.sum(g) == 0:
				print(x)
				break
			d = self.dk(H,g)
			alpha = self.alpha_k(g,d,self.Q)
			x = x + alpha * d

			del_x = alpha * d
			g_new = np.dot(self.Q, x)
			del_g = g_new - g
			g = g_new 
			
			temp = del_x - np.dot(H,del_g)
			H = H + np.dot(temp,temp.transpose())/np.dot(del_g.transpose(),temp)



if __name__ == '__main__':
	Q = np.array([[2,0],[0,1]])

	X0 = np.array([[1],[2]])

	rank_one = Rank_One(Q,X0)
	rank_one.func()

