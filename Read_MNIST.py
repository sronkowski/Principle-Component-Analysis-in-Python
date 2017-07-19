#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:58:14 2017

@author: SwatzMac

@Program: PCA on MNIST Data
"""

import os, struct
#import numpy
import matplotlib as plt
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros as np
import numpy.linalg as LA

#numpy.set_printoptions(threshold=numpy.inf)
def load_mnist(dataset="training", digits=range(10), 
   path='/Users/SwatzMac/Documents/Study/Classes/Machine Learning, Statistics and Python/Python_Programs/PCA'):
    
    

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images-idx-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]       
    N = len(ind)
    print('N =', N)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

    
    
from pylab import *
from numpy import *
import scipy.sparse as sparse
import scipy.linalg as linalg

images, labels = load_mnist('training', digits=[6,9])

# converting from NX28X28 array into NX784 array
flatimages = list()
for i in images:
    flatimages.append(i.ravel())
X = asarray(flatimages)
#print(X)
'''
print("Check shape of matrix", X.shape)
print(X.shape)
print("Check Mins and Max Values",np.amin(X),np.amax(X))
print("\nCheck training vector by plotting image \n")
plt.imshow(X[20].reshape(28, 28),interpolation='None', cmap=cm.gray)
show()'''

print('N: \n',len(flatimages))

# XZCVPR Process
print ('X Value\n', X)

#Mu = mean(X, axis=0)
##X.mean(axis = 0)
#print('Mu Values\n', Mu)
#
#thefile = open('MuText.txt','w')
#for item in Mu:
#    thefile.write("%s," % item)
#
#    
#thefile.close()
T = labels.flatten()
[[labeln, labelp], [Nn, Np]] = unique(T, return_counts = True)

print('Target: ',T)
print('lablen: ',labeln)
print('labelp: ',labelp)
print('Nn: ',Nn)
print('Np: ',Np)

for i in X:
    for j in labeln: 
        Mun = mean(Xn, axis=0)
print (Mun)


X = X[:,0:2]
print('X shape: \n',X.shape)
print(X)

Mu = mean(X, axis=0)
##X.mean(axis = 0)
#print('Mu shape: \n', Mu.shape)

Z = X - Mu
#print('Z Values\n', Z)
print ('Mu Size: \n',Mu.shape)

C = cov(Z,rowvar = False)
#print('C Values\n', C)

[L,V] = LA.eigh(C)
#print('L Values\n', L, '\n\n','V Values\n', V)
print(V.shape)

'''
Let us examine the V matrix of EigenVectors. Do the rows represent eigenvectors? or 
do the columns represent eigenvectors? 
'''

row = V[0,:]
col = V[:,0]

col_ans = dot(C,col)/(L[0]*col)
#print(col_ans)

row_ans = dot(C,row)/(L[0]*row)
#print(row_ans)
#print(np.dot(C,row)/(L[0]*row)) #If the matrix product C.row is the same as L[0]*row, the 
#result should be [1,1,1...]
#print(np.dot(C,col)/(L[0]*col)) #If the matrix product C.col is the same as L[0]*col, 
# the result should [1,1,1...]

# So we conclude that the columns of V are eigenvectors. 

L = flipud(L)
V = flipud(V.T)
V = V[:2] #work with 2 eigenvectors
print ('V - 2eigenvectors: \n',V)

#V1file = open('V1Text.txt','w')
#for item in V[0]:
#    V1file.write("%s," % item)
#
#V1file.close()
#
#V2file = open('V2Text.txt','w')
#for item in V[1]:
#    V2file.write("%s," % item)
#
#V2file.close()


row = V[0,:] #check once again
#print(dot(C,row)/(L[0]*row)) #If the matrix product C.row is same as L[0]*row, result should be  [1,1,1,...]

print('Norm V[0] = ', linalg.norm(V[0]))
print('Norm V[1] = ', linalg.norm(V[1]))
print('Orthogonality = ', dot(V[0,:],V[1,:]))

P = dot(Z,V.T)
print('P Shape= ', P.shape)
Mu2D = mean(X,axis=0)
print ('Mean Vector 2D = ',Mu2D)

##2D Approximation
#P2D = P[0:2,:]
#Mu2D = Mu[0:2,:]
#V2D = V[:,0:2,:]
#print('P shape =', P.shape)
#print('P2D shape =', P2D.shape)
#print('Mu shape =', Mu.shape)
#print('Mu2D shape =', Mu2D.shape)
#print('V2D shape =', V2D.shape)

#print('2D P =',P[0:2,:])
##print ('P Values\n',P)
#print (P[0,0],P[0,1],P[1,0],P[1,1])

#print ('2D Mu = ',Mu[0:2,:])

R = dot(P,V)
print('R Shape= ', R.shape)

#print ('R - Z Values \n',R-Z) #Z is recovered since R-Z is seen to contain very small values

Xrec = R + Mu
#print('Xrec - X Values\n',Xrec - X) #X is recovered since Xrec-X is seen to contain very small values
print('Xrec Shape= ', Xrec.shape)

Xrec1 = (dot(P[:,0:1],V[0:1,:])) + Mu
#print (Xrec1)
print('Xrec1 Shape= ', Xrec1.shape)

Xrec2 = (dot(P[:,0:2],V[0:2,:])) + Mu
print('Xrec2 Shape= ', Xrec2.shape)


#index=20
#
#plt.imshow(X[index].reshape(28, 28),interpolation='None', cmap=cm.gray)
#show()
#
#plt.imshow(Xrec[index].reshape(28, 28),interpolation='None', cmap=cm.gray)
#show()
#
#plt.imshow(Xrec1[index].reshape(28, 28),interpolation='None', cmap=cm.gray)
#show()
#
#plt.imshow(Xrec2[index].reshape(28, 28),interpolation='None', cmap=cm.gray)
#show()

#print('Xrec-X values\n',Xrec1-X)
#print(X[1:2])
