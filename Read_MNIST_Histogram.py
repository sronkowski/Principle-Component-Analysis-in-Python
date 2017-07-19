#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:58:14 2017

@author: SwatzMac

@Program: PCA on MNIST Data
"""

import os, struct
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

    
import numpy
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

# XZCVPR Process
#print ('X Value\n', X)

#thefile = open('Mu.txt', 'w')

Mu = mean(X, axis=0)
#X.mean(axis = 0)
#print('Mu Values\n', Mu)
#
#for item in Mu:
#  thefile.write("%s," % item)
#
#thefile.close()

Z = X - Mu
print('Z Values\n', Z)

C = cov(Z,rowvar = False)
print('C Values\n', C)

[L,V] = LA.eigh(C)
#print('L Values\n', L, '\n\n','V Values\n', V)
print('V shape: ',V.shape)

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
row = V[0,:] #check once again
#print(dot(C,row)/(L[0]*row)) #If the matrix product C.row is same as L[0]*row, result should be  [1,1,1,...]

print('V Shape with 2 eigenvectors= ', V.shape)

#v1File = open("v1file.txt", "w")
#for item in V[0]:
#  v1File.write("%s," % item)
#v1File.close()
#
#v2File = open("v2file.txt", "w")
#for item in V[1]:
#  v2File.write("%s," % item)
#v2File.close()

#Checking Normalization and Orthogonality of eigenvectors
print('Norm V[0] = ', linalg.norm(V[0]))
print('Norm V[1] = ', linalg.norm(V[1]))
print('Orthogonality = ', dot(V[0,:],V[1,:]))

P = dot(Z,V.T)
print('P Shape= ', P.shape)
print('MeanP = ', mean(P,axis=0))

#PFile = open("Pfile.txt", "w")
#for item in P:
#  PFile.write("%s \n"  % item)
#PFile.close()

R = dot(P,V)
print('R Shape= ', R.shape)

#print ('R - Z Values \n',R-Z) #Z is recovered since R-Z is seen to contain very small values

Xrec = R + Mu
#print('Xrec - X Values\n',Xrec - X) #X is recovered since Xrec-X is seen to contain very small values
#print('Xrec Shape= ', Xrec.shape)

Xrec1 = (dot(P[:,0:1],V[0:1,:])) + Mu
#print (Xrec1)
#print('Xrec1 Shape= ', Xrec1.shape)

Xrec2 = (dot(P[:,0:2],V[0:2,:])) + Mu
#print('Xrec2 Shape= ', Xrec2.shape)

index=20
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

#------------------------------------------------------------------------

T = labels.flatten()

PositiveP = list()
NegativeP = list()

PositiveX = list()
NegativeX = list()

PositiveZ = list()
NegativeZ = list()

PositiveR = list()
NegativeR = list()

PositiveXREC = list()
NegativeXREC = list()


for i in range(len(T)):
    if (T[i] == 6):
        PositiveP.append(P[i])
        PositiveX.append(X[i])
        PositiveZ.append(Z[i])
        PositiveR.append(R[i])
        PositiveXREC.append(Xrec[i])
        
    elif (T[i] == 9):
        NegativeP.append(P[i])
        NegativeX.append(X[i])
        NegativeZ.append(Z[i])
        NegativeR.append(R[i])
        NegativeXREC.append(Xrec[i])

print (len(PositiveP))
print (len(NegativeP))

PosP = asarray(PositiveP)
NegP = asarray(NegativeP)

PosX = asarray(PositiveX)
NegX = asarray(NegativeX)

PosZ = asarray(PositiveZ)
NegZ = asarray(NegativeZ)

PosXREC = asarray(PositiveXREC)
NegXREC = asarray(NegativeXREC)

PosR = asarray(PositiveR)
NegR = asarray(NegativeR)

print ('PosP Shape = ', PosP.shape)
print ('NegP Shape = ', NegP.shape)

MuP = mean(PosP, axis=0)
MuN = mean(NegP, axis=0)

print ('MuP Shape = ', MuP.shape)
print ('MuN Shape = ', MuN.shape)

print ('MuP = ', MuP)
print ('MuN = ', MuN)

ZP = PosP - MuP
ZN = NegP - MuN

CP = cov(ZP,rowvar = False)
print ('CP = ', CP)

CN = cov(ZN,rowvar = False)
print ('CN = ', CN)

min_1stP = min(P[:,0])
max_1stP = max(P[:,0])

min_2ndP = min(P[:,1])
max_2ndP = max(P[:,1])

print ('min_1stP = ', min_1stP)
print ('max_1stP = ', max_1stP)
print ('min_2ndP = ', min_2ndP)
print ('max_2ndP = ', max_2ndP)


# Build the 2D Scatter Plot
import matplotlib.pyplot as plt
import matplotlib.pyplot as mplot

#PosP Shape =  (5918, 2)
#NegP Shape =  (5949, 2)


plt.scatter(PosP[0:5918,:],NegP[0:5918,:],color=['blue','red'],alpha=0.25)
plt.show()

#x = PosP[0:5918,:2]
#y = NegP[0:5918,:2]

#mplot.scatter(x, y, color=['blue','red'])
#mplot.scatter(x,y)
#plt.show()

# Constructing 2D Histograms 
#define Bin Size 
B = 25

# Compute min and max of 2 principle components 
Pmin1 = min(P[:,0])
Pmax1 = max(P[:,0])

Pmin2 = min(P[:,1])
Pmax2 = max(P[:,1])

#Initialize Histogram to Zeroes 
Hp = np.zeros((B,B))
Hn = np.zeros((B,B))
#print (Hp, Hn)

P6 = PosP[:,0]
P6_2 = PosP[:,1]
P9 = NegP[:,0]
P9_2 = NegP[:,1]

P1 = P[:,0]
P2 = P[:,1]

for (T,p1,p2) in zip(T,P1,P2):
    
     row = int((B-1)*((p1-Pmin1)/(Pmax1-Pmin1)))
     col = int((B-1)*((p2-Pmin2)/(Pmax2-Pmin2)))
     if T == 6:
         Hp[row,col] += 1
     if T == 9:
         Hn[row,col] += 1

#     
#print ('H1: \n',Hp)
#print ('H2: \n',Hn)

#for (T,Pp,Pn) in zip(T,P6,P6_2):
#     #print (T)
#    #print (T,P6,P6_2)
#    #print (Pmin1, Pmax1, Pmin2, Pmax2)
#    
#     row = int((B-1)*((Pp-Pmin1)/(Pmax1-Pmin1)))
#     col = int((B-1)*((Pn-Pmin2)/(Pmax2-Pmin2)))
#     Hp[row,col] += 1

#print ('H1: \n',Hp)


#for (Pp,Pn) in zip(P9,P9_2):
#     #print (T)
#    #print (T,P6,P6_2)
#    #print (Pmin1, Pmax1, Pmin2, Pmax2)
#    
#     row = int((B-1)*((Pp-Pmin1)/(Pmax1-Pmin1)))
#     col = int((B-1)*((Pn-Pmin2)/(Pmax2-Pmin2)))
#     Hn[row,col] += 1
#
#
#
#------------Writing out Histogram outputs to Two Files
#
#Hist_File1 = open("hist1_file.txt", "w")
#for i in range(B):
#    for j in range(B):
#        Hist_File1.write("%d " % Hp[i][j])
#    Hist_File1.write("\n")
#Hist_File1.close()
#
#Hist_File2 = open("hist2_file.txt", "w")
#for i in range(B):
#    for j in range(B):
#        Hist_File2.write("%d " % Hn[i][j])
#    Hist_File2.write("\n")
#Hist_File2.close()

#==============================================================================
# for i in range(len(T)):
#     if (T[i] == 6):
#         PositiveP.append(P[i])
#     elif (T[i] == 9):
#         NegativeP.append(P[i])
#==============================================================================
        
        
PosRand = 2851 #randint(0,len(PosP))
NegRand = 4169 #randint(0,len(NegP))

print('PosRand = ', PosRand)
print('NegRand = ', NegRand)

Xp = PosX[PosRand] 
#XpFile = open("xp_file.txt", "w")
#for item in Xp:
#  XpFile.write("%s," % item)
#XpFile.close()

Xn = NegX[NegRand] 
#XnFile = open("xn_file.txt", "w")
#for item in Xn:
#  XnFile.write("%s," % item)
#XnFile.close()

Zpp = PosZ[PosRand]
Znn = NegZ[NegRand]

#ZFile = open("z_file.txt", "w")
#for item in Zpp:
#  ZFile.write("%s," % item)
#ZFile.write("\n\n\n")   
#for item2 in Znn:
#  ZFile.write("%s," % item2)  
#ZFile.close()

Rpp = PosR[PosRand]
Rnn = NegR[NegRand]

#==============================================================================
# RFile = open("r_file.txt", "w")
# for item in Rpp:
#   RFile.write("%s," % item)
# RFile.write("\n\n\n")   
# for item2 in Rnn:
#   RFile.write("%s," % item2)  
# RFile.close()
#==============================================================================


XRECpp = PosXREC[PosRand]
XRECnn = NegXREC[NegRand]

#==============================================================================
# XRECFile = open("XREC_file.txt", "w")
# for item in XRECpp:
#   XRECFile.write("%s," % item)
# XRECFile.write("\n\n\n")   
# for item2 in XRECnn:
#   XRECFile.write("%s," % item2)  
# XRECFile.close()
#==============================================================================

Ppppp = PosP[PosRand] 
Nnnnn = NegP[NegRand]

#==============================================================================
# PPPPFile = open("PPPPFile.txt", "w")
# for item in Ppppp:
#   PPPPFile.write("%s," % item)
# PPPPFile.write("\n\n\n")   
# for item2 in Nnnnn:
#   PPPPFile.write("%s," % item2)  
# PPPPFile.close()
#==============================================================================

print ('Shape of Xp: ',Xp.shape)
print ('Shape of Xn: ',Xn.shape)


#################### Bayesian Classifier ##################


#==============================================================================
# def norm_pdf_multivariate(x, mu, sigma):
#     size = len(x)
#     if size == len(mu) and (size, size) == sigma.shape:
#         det = linalg.det(sigma)
#         if det == 0:
#             raise NameError("The covariance matrix can't be singular")
# 
#             norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2))
#             x_mu = matrix(x - mu)
#             inv = sigma.I        
#             result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
#             return norm_const * result
#     else:
#         raise NameError("The dimensions of the input don't match")
#==============================================================================
        
#==============================================================================
# def Bayesian (x,mean,stdev):
#     exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
#     return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
#==============================================================================

#==============================================================================
# print (len(Xp3))
# print (Xp.shape)
# print (PosX.shape)
# print (StdP.shape)
# print (len(Mu))
#==============================================================================


# BAYESIAN CLASSIFIER
#==============================================================================
def pdf_multivariate_gauss(x, mu, cov):
    '''
    Caculate the multivariate normal density (pdf)

    Keyword arguments:
        x = numpy array of a "d x 1" sample vector
        mu = numpy array of a "d x 1" mean vector
        cov = "numpy array of a d x d" covariance matrix
    '''
    '''
    assert(mu.shape[0] > mu.shape[1]), 'mu must be a row vector'
    assert(x.shape[0] > x.shape[1]), 'x must be a row vector'
    assert(cov.shape[0] == cov.shape[1]), 'covariance matrix must be square'
    assert(mu.shape[0] == cov.shape[0]), 'cov_mat and mu_vec must have the same dimensions'
    assert(mu.shape[0] == x.shape[0]), 'mu and x must have the same dimensions'
    '''
    part1 = 1 / ( ((2* np.pi)**(len(mu)/2)) * (np.linalg.det(cov)**(1/2)) )
    part2 = (-1/2) * ((x-mu).T.dot(np.linalg.inv(cov))).dot((x-mu))
    return float(part1 * np.exp(part2))

muup = mean(PosP, axis = 0)
stdep = std(PosP, axis = 0)
covvp = cov(PosP, rowvar = False)

muun = mean(NegP, axis = 0)
stden = std(NegP, axis = 0)
covvn = cov(NegP, rowvar = False)

Xpos = PosP[PosRand]
valuep = pdf_multivariate_gauss(Xpos,muup,covvp)
valuen = pdf_multivariate_gauss(Xpos,muun,covvn)
denom = (len(PosP) * valuep) + (len(NegP) * valuen)
Prob_Xp = (len(PosP) * valuep) / denom

Xneg = NegP[NegRand]
valuepp = pdf_multivariate_gauss(Xneg,muup,covvp)
valuenn = pdf_multivariate_gauss(Xneg,muun,covvn)
denom = (len(PosP) * valuepp) + (len(NegP) * valuenn)
Prob_Xn = (len(NegP) * valuenn) / denom

print (Prob_Xp, Prob_Xn) #Probability as a result of classifying xn and xp using Bayesian

btp = 0
btn = 0
bfp = 0
bfn = 0

for i in range(len(PosP)):
    xpos = PosP[i]
    valuepi = pdf_multivariate_gauss(xpos,muup,covvp)
    valueni = pdf_multivariate_gauss(xpos,muun,covvn)
    denom = (len(PosP) * valuepi) + (len(NegP) * valueni)
    probbb = (len(PosP) * valuepi) / denom
    if (probbb > 0.5):
        btp += 1
    else:
        bfn += 1
        
for j in range(len(NegP)):
    xneg = NegP[j]
    valuepi = pdf_multivariate_gauss(xneg,muup,covvp)
    valueni = pdf_multivariate_gauss(xneg,muun,covvn)
    denom = (len(PosP) * valuepi) + (len(NegP) * valueni)
    probbb = (len(NegP) * valueni) / denom
    if (probbb > 0.5):
        btn += 1
    else:
        bfp += 1

bPrecision = btp / (btp + bfp)
bRecall = btp / (btp + bfn)
bAccuracy = (btp + btn)/(btp + btn + bfp + bfn)

print ('Bayesian Tests = ',bPrecision, bRecall, bAccuracy)


####### HISTOGRAM CLASSIFIER #########

xp1 = P6[PosRand]
xp2 = P6_2[PosRand]
Prow_Xp = int((B-1)*((xp1-Pmin1)/(Pmax1-Pmin1)))
Pcol_Xp = int((B-1)*((xp2-Pmin2)/(Pmax2-Pmin2)))
Pbin_xp = Hp[Prow_Xp][Pcol_Xp]
Nbin_xp = Hn[Prow_Xp][Pcol_Xp]
Prob_Xp = (float) (Pbin_xp / (Pbin_xp + Nbin_xp))
print(Pbin_xp, Nbin_xp, Prob_Xp) 


xn1 = P9[NegRand]
xn2 = P9_2[NegRand]
Nrow_Xn = int((B-1)*((xn1-Pmin1)/(Pmax1-Pmin1)))
Ncol_Xn = int((B-1)*((xn2-Pmin2)/(Pmax2-Pmin2)))
Pbin_xn = Hp[Nrow_Xn][Ncol_Xn]
Nbin_xn = Hn[Nrow_Xn][Ncol_Xn]
Prob_Xn = (float) (Nbin_xn / (Nbin_xn + Pbin_xn))
print(Pbin_xn, Nbin_xn, Prob_Xn)


#print(PosP.shape)
tp = 0
tn = 0
fp = 0
fn = 0

for i in range(len(PosP)):
    xp1 = P6[i]
    xp2 = P6_2[i]
    Prow_Xp = int((B-1)*((xp1-Pmin1)/(Pmax1-Pmin1)))
    Pcol_Xp = int((B-1)*((xp2-Pmin2)/(Pmax2-Pmin2)))
    Pbin_xp = Hp[Prow_Xp][Pcol_Xp]
    Nbin_xp = Hn[Prow_Xp][Pcol_Xp]
    Prob_Xp = (float) (Pbin_xp / (Pbin_xp + Nbin_xp))
    if (Prob_Xp > 0.5):
        tp += 1
    else:
        fn += 1

for j in range(len(NegP)):
    xn1 = P9[j]
    xn2 = P9_2[j]
    Nrow_Xn = int((B-1)*((xn1-Pmin1)/(Pmax1-Pmin1)))
    Ncol_Xn = int((B-1)*((xn2-Pmin2)/(Pmax2-Pmin2)))
    Pbin_xn = Hp[Nrow_Xn][Ncol_Xn]
    Nbin_xn = Hn[Nrow_Xn][Ncol_Xn]
    Prob_Xn = (float) (Nbin_xn / (Nbin_xn + Pbin_xn))
    
    if (Prob_Xn > 0.5):
        tn += 1
    else:
        fp += 1
        

Precision = tp / (tp + fp)
Recall = tp / (tp + fn)
Accuracy = (tp + tn)/(tp + tn + fp + fn)

print ('Histogram Tests = ', Precision, Recall, Accuracy)




# Finding Accuracy of Classifiers
#==============================================================================
# import numpy as np
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import accuracy_score
# 
# T = labels.flatten()
# # Input training data
# training_points = np.array(X)
# training_labels = np.array(T)
# X = np.array(training_points)
# Y = np.array(training_labels)
# print(training_points.shape)
# print(training_labels)
# # Create Naive Bayes classifier
# clf = GaussianNB()
# clf.fit(X, Y)
# 
# # Classify test data with the classifier
# #test_points = [[1, 1], [2, 2], [3, 3], [4, 3]]
# #test_labels = [2, 2, 2, 1]
# predicts = clf.predict(training_points)
# 
# # Calculate Accuracy Rate manually
# #count = len(["ok" for idx, label in enumerate(test_labels) if label == predicts[idx]])
# #print "Accuracy Rate, which is calculated manually is: %f" % (float(count) / len(test_labels))
# 
# # Calculate Accuracy Rate by using accuracy_score()
# print ("Accuracy Rate, which is calculated by accuracy_score() is: %f" % accuracy_score(training_points, predicts))
# 
#==============================================================================
