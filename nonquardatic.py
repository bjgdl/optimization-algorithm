#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 17:35:00 2019

@author: gaudel
"""

import numpy as np

#expression
fun = lambda x:100*(x[0]**2 - x[1]**2)**2 +(x[0] - 1)**2
# fun = lambda x:x[0]**2+0.5*x[1]**2+3

#gradient vector
gfun = lambda x:np.array([400*x[0]*(x[0]**2 - x[1]) + 2*(x[0] - 1),-200*(x[0]**2 - x[1])])

#Hessian matrix
hess = lambda x:np.array([[1200*x[0]**2 - 400*x[1] + 2,-400*x[0]],[-400*x[0],200]])

def rank1(fun,gfun,hess,x0):
    maxk = 1e5
    rho = 0.55
    sigma = 0.4
    epsilon = 1e-5
    k = 0
    n = np.shape(x0)[0]

    Bk = np.eye(2)

    while k < maxk:
        gk = gfun(x0)
        if np.linalg.norm(gk) < epsilon:
            break
        dk = -1.0 * np.linalg.solve(Bk, gk)
        m = 0
        mk = 0
        while m < 20:
            if fun(x0 + rho ** m * dk) < fun(x0) + sigma * rho ** m * np.dot(gk, dk):
                mk = m
                break
            m += 1


        x = x0 + rho ** mk * dk
        sk = rho**mk*dk
        yk = gfun(x) - gk

        if np.dot(sk, yk) > 0:
            Bs = np.dot(Bk, yk)
            Bk = Bk + np.dot((sk-Bs),(sk-Bs))/np.dot(yk,(sk-Bs))

        k += 1
        x0 = x
        return x0,fun(x0),k



def bfgs(fun,gfun,hess,x0):
    maxk = 1e5
    rho = 0.55
    sigma = 0.4
    epsilon = 1e-5
    k = 0
    n = np.shape(x0)[0]

    Bk = np.eye(2)
    while k < maxk:
        gk = gfun(x0)
        if np.linalg.norm(gk) < epsilon:
            break
        dk = -1.0*np.linalg.solve(Bk,gk)
        m = 0
        mk = 0
        while m < 20:
            if fun(x0+rho**m*dk) < fun(x0)+sigma*rho**m*np.dot(gk,dk):
                mk = m
                break
            m += 1


        x = x0 + rho**mk*dk
        print("the" + str(k) + "iteartion：" + str(x))
        sk = x - x0
        yk = gfun(x) - gk

        if np.dot(sk,yk) > 0:
            Bs = np.dot(Bk,sk)
            ys = np.dot(yk,sk)
            sBs = np.dot(np.dot(sk,Bk),sk)

            Bk = Bk - 1.0*Bs.reshape((n,1))*Bs/sBs + 1.0*yk.reshape((n,1))*yk/ys

        k += 1
        x0 = x

    return x0,fun(x0),k


def dfp(fun,gfun,hess,x0):

    maxk = 1e5
    rho = 0.05
    sigma = 0.4
    epsilon = 1e-5
    k = 0
    n = np.shape(x0)[0]

    Hk = np.eye(2)

    while k < maxk:
        gk = gfun(x0)
        if np.linalg.norm(gk) < epsilon:
            break
        dk = -1.0*np.dot(Hk,gk)


        m = 0
        mk = 0
        while m < 20:
            if fun(x0 + rho**m*dk) < fun(x0) + sigma*rho**m*np.dot(gk,dk):
                mk = m
                break
            m += 1

        x = x0 + rho**mk*dk
        print ("the"+str(k)+"iteration："+str(x))
        sk = x - x0
        yk = gfun(x) - gk

        if np.dot(sk,yk) > 0:
            Hy = np.dot(Hk,yk)
            sy = np.dot(sk,yk)
            yHy = np.dot(np.dot(yk,Hk),yk)
            Hk = Hk - 1.0*Hy.reshape((n,1))*Hy/yHy + 1.0*sk.reshape((n,1))*sk/sy

        k += 1
        x0 = x
    return x0,fun(x0),k

x0 ,fun0 ,k = rank1(fun,gfun,hess,np.array([0,0]))
print("total iterations are",k,"  Minimun point:",x0)