#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 04:22:19 2019

@author: gaudel
"""



import numpy as np 

class BFGS():

	def __init__(self,Q,b,X0):
		self.Q = Q
		self.X0 = X0
		self.b = b
		self.H0 = np.eye(X0.size)

	def gk(self,Q,X):
		gk = np.dot(Q,X)-self.b
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
			g_new = np.dot(self.Q, x)-self.b
			del_g = g_new - g
			g = g_new 
			print(g)
			
			print(del_x)
			print(del_g)
			temp1=(1+(np.dot(np.dot(del_g.transpose(),H),del_g)/np.dot(del_g.transpose(),del_x)))
			temp2=(np.dot(del_x,del_x.transpose())/np.dot(del_x.transpose(),del_g))
			temp3 = np.dot(np.dot(del_x,del_g.transpose()),H) + np.dot(np.dot(H,del_g),del_x.transpose())

			temp3 = temp3/np.dot(del_g.transpose(),del_x)
			# print(temp1)
			# print(temp2)
			# print(temp3)
			H = H+temp1*temp2-temp3
			print(H)
			


if __name__ == '__main__':
	Q = np.array([[5,-3],[-3,2]])
	b= np.array([[0],[1]])

	X0 = np.array([[0],[0]])

	rank_one = BFGS(Q,b,X0)
	rank_one.func()

