#!/usr/bin/env python
import matplotlib.pyplot as plt
import matplotlib
import numpy as np 
import seaborn as sns


def makeplot(func,xmin,xmax):

    x,y,derv,names = func(xmin,xmax)

    fig,axis = plt.subplots()
    axis.plot(x,y,color=sns.xkcd_rgb['merlot'])
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    axis.set_title(names[0])

    fig2,axis2 = plt.subplots()
    axis2.plot(x,derv,color=sns.xkcd_rgb['merlot'])
    axis2.set_xlabel('x')
    axis2.set_ylabel(r'\frac{dy}{dx}')
    axis2.set_title(names[1])

    fig.savefig('./figs/{n}.pdf'.format(n=names[2]))
    fig2.savefig('./figs/{n}_derivative.pdf'.format(n=names[2]))


def act_sigmoid(xmin,xmax):
    x = np.linspace(xmin,xmax,100)
    y = 1/(1 + np.exp(-x))
    derv = y*(1-y)
    names = (r'y = \sigma(x)',r'\frac{d\sigma(x)}{dx}','sigmoid')
    return x,y,derv,names

def act_tanh(xmin,xmax):
    x = np.linspace(xmin,xmax,100)
    y = np.tanh(x)
    derv = 1-np.power(y,2)
    names = (r'y = tanh(x)',r'1 - tanh^2(x)','tanh')
    return x,y,derv,names

def act_ReLu(xmin,xmax):
    x = np.linspace(xmin,xmax,100)
    y = np.where(x >= 0,x,0)
    derv = np.where(x>=0,1,0)
    names = (r'y = 0 (x< 0) , x (x >= 0)',r'0 (x <0) , 1 (x >=0)','ReLu')
    return x,y,derv,names

def act_leaky_ReLu(xmin,xmax):
    x = np.linspace(xmin,xmax,100)
    y = np.where(x >= 0,x,0.01*x)
    derv = np.where(x>=0,1,0.01)
    names = (r'y = 0.01x (x< 0) , x (x >= 0)',r'0.01 (x <0) , 1 (x >=0)','leaky ReLu')
    return x,y,derv,names


def GeneratePlots():

    makeplot(act_sigmoid,-10,10)
    makeplot(act_tanh,-10,10)
    makeplot(act_ReLu,-10,10)
    makeplot(act_leaky_ReLu,-10,10)

if __name__ == '__main__':
    sns.set()
    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif')
    GeneratePlots()