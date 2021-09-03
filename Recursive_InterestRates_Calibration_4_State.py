#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 11:59:13 2021

@author: gayanithalagoda
"""


import numpy as np
import pandas as pd
from random import seed
from random import random
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
import math 
import random
import scipy
import itertools

z = pd.read_csv("/Users/gayanithalagoda/Desktop/masters thesis/MASTERS_RESEARCH_CODE/Research_Content/Datasets/Simulated_data/ytt2.csv")
z = np.array(z[0:999])
intrate=z
n= 60
state = 4
batch = 60
no_passes = 16

alpha=[0.99, 0.98, 0.67,0.67 ]
gamma = [0.0078, 0.0048,0.0048,0.0067 ]
nu = [1- 3*0.0105, 1- 6*0.0105, 1- 6*0.0105,1- 6*0.0105]
fish_alpha = [0,0,0,0]
fish_gamma=[0,0,0,0]
fish_nu=[0,0,0,0]

Pi = np.array([[1/4,1/4,1/4,1/4],[1/4,1/4,1/4,1/4],[1/4,1/4,1/4,1/4],[1/4,1/4,1/4,1/4]])

X=np.array([1,0,0,0])
alphaIP=0
gammaIP=0
nuIP=0

for i in range(0,state):
    a1= alpha[i]*X[i]
    alphaIP= alphaIP + a1
    b1=gamma[i]*X[i]
    gammaIP=gammaIP + b1
    c1 = nu[i]*X[i]
    nuIP = nuIP + c1

#z(1)=0;
lambda1 = np.exp(-(alphaIP*intrate[0]+gammaIP)*intrate[1]/(nuIP**2) - 0.5*((alphaIP*intrate[0]+gammaIP)/nuIP)**2)
Loglikelihood1= -0.5*np.log(2*math.pi*nuIP**2)- (intrate[0]-alphaIP-gammaIP)**2/(2*nuIP**2)

X2=X
sigmaJ1 = np.zeros([state,state])
sigmaJ2 = np.zeros([state,state])
sigmaJ3 = np.zeros([state,state])
sigmaJ4 = np.zeros([state,state])
sigmaO = np.zeros([state,state])
sigmaT1 = np.zeros([state,state])
sigmaT2 = np.zeros([state,state])
sigmaT3 = np.zeros([state,state])
for i in range(0,state):
    sigmaJ1[:,i]= lambda1 * X[i]*X[0]*X2
    sigmaJ2[:,i]= lambda1 * X[i]*X[1]*X2
    sigmaJ3[:,i]= lambda1 * X[i]*X[2]*X2
    sigmaJ4[:,i]= lambda1 * X[i]*X[3]*X2
    sigmaO[:,i]= lambda1 * X[i]*X2
    sigmaT1[:,i] = lambda1 * X[i] * intrate[0]* X2
    sigmaT2[:,i] = lambda1 * X[i] * (intrate[0]**2)* X2
    sigmaT3[:,i] = lambda1 * X[i] * (intrate[0]*intrate[1])* X2
E=np.identity(4)
inta = 2 # 2
inte = 3 + batch # 3+ batch
sigmaX = np.zeros([state,no_passes*batch+2])#np.zeros([state,batch+2])
sumX = np.zeros(no_passes*batch+2)#np.zeros(batch+2)
sigmaX[:,0] = X
sumX[0] = sum(sigmaX[:,0])

Gamma = np.zeros([batch+1,state])
sigmaXk = np.zeros([state,no_passes*batch+2])#np.zeros([state,batch+2])
sumJ1 = np.zeros([state,state]) 
sumJ2 = np.zeros([state,state])
sumJ3 = np.zeros([state,state])
sumJ4 = np.zeros([state,state])
remainderSJ1 = np.zeros([state,no_passes*batch+2])#np.zeros([state,batch+2]) #3 * 21
remainderSJ2 = np.zeros([state,no_passes*batch+2])#np.zeros([state,batch+2])
remainderSJ3 = np.zeros([state,no_passes*batch+2])
remainderSJ4 = np.zeros([state,no_passes*batch+2])#np.zeros([state,batch+2])
remainderSO =  np.zeros([state,no_passes*batch+2])#np.zeros([state,batch+2])
remainderST1 = np.zeros([state,no_passes*batch+2])#np.zeros([state,batch+2])
remainderST2 = np.zeros([state,no_passes*batch+2])#np.zeros([state,batch+2])
remainderST3 = np.zeros([state,no_passes*batch+2])#np.zeros([state,batch+2])
sumO = np.zeros([state,state])
sumT1=np.zeros([state,state])
sumT2=np.zeros([state,state])
sumT3=np.zeros([state,state])
a= np.zeros(state)
beta=np.zeros(state)
theta=np.zeros(state)
intr = np.zeros(batch+1)
Loglikelihood=np.zeros([batch,1])
LogLike =np.zeros([batch,no_passes])
F=np.zeros([batch,no_passes])
AL= np.zeros([no_passes,state])
GA= np.zeros([no_passes,state])
NU= np.zeros([no_passes,state])
FISH_AL= np.zeros([no_passes,state])
FISH_GA= np.zeros([no_passes,state])
FISH_NU= np.zeros([no_passes,state])
PROB1 = np.zeros([no_passes,1])
PROB2 = np.zeros([no_passes,1])
PROB3 = np.zeros([no_passes,1])
PROB4 = np.zeros([no_passes,1])
PROB5 = np.zeros([no_passes,1])
PROB6 = np.zeros([no_passes,1])
PROB7 = np.zeros([no_passes,1])
PROB8 = np.zeros([no_passes,1])
PROB9 = np.zeros([no_passes,1])
PROB10 = np.zeros([no_passes,1])
PROB11 = np.zeros([no_passes,1])
PROB12 = np.zeros([no_passes,1])
PROB13 = np.zeros([no_passes,1])
PROB14 = np.zeros([no_passes,1])
PROB15 = np.zeros([no_passes,1])
PROB16 = np.zeros([no_passes,1])

LogLikelihoodTotal = np.zeros(no_passes)
AIC3st=np.zeros(no_passes)
BIC3st=np.zeros(no_passes)
#PROB=np.zeros(9)

for u in range(0,no_passes):
    IR=intrate[inta-1:inte] # from 2nd observation to 22.  # check again
    N= len(IR) #22
    for k in range(0,batch+1):#batch+1
        Xk = sigmaX[:,inta+k-2]
        for i in range(0,state):
            Gamma[k,i]=math.exp(-(IR[k+1]*(alpha[i]*IR[k]+gamma[i])/nu[i]**2) - ((alpha[i]*IR[k]+gamma[i])**2)/(2*nu[i]**2))
        sigmaXk[:,inta+k-1]  = Gamma[k,0] *sigmaX[0,inta+k-2]*Pi[:,0] + Gamma[k,1] * sigmaX[1,inta+k-2]*Pi[:,1] + Gamma[k,2] * sigmaX[2,inta+k-2]*Pi[:,2] + Gamma[k,3] * sigmaX[3,inta+k-2]*Pi[:,3]
        sumX[inta+k-1]= np.sum(sigmaXk[:,inta+k-1]) #total probability theory used
        sigmaX[:,inta+k-1]=sigmaXk[:,inta+k-1]/sumX[inta+k-1] #this chunk correct , make sure that sigmaX row sum=1 always

        for r in range(0,state):
            sumJ1[:,r]= Gamma[k,0]*sigmaJ1[0,r]*Pi[:,0]+ Gamma[k,1] *sigmaJ1[1,r]*Pi[:,1] + Gamma[k,2]*sigmaJ1[2,r]*Pi[:,2] + Gamma[k,3]*sigmaJ1[3,r]*Pi[:,3] 
            sigmaJ1[:,r]= sumJ1[:,r]+ Gamma[k,r]*sigmaX[r,inta+k-2]*Pi[0,r]*E[:,0]
        sumJ1a=sum(sigmaJ1)/sumX[inta+k-2]
        remainderSJ1[:,inta+k-1]=sumJ1a.transpose()

        for r in range(0,state):
            sumJ2[:,r]= Gamma[k,0] *sigmaJ2[0,r]*Pi[:,0]+Gamma[k,1] *sigmaJ2[1,r]*Pi[:,1] + Gamma[k,2]*sigmaJ2[2,r]*Pi[:,2]+  Gamma[k,3]*sigmaJ2[3,r]*Pi[:,3] 
            sigmaJ2[:,r]=sumJ2[:,r]+Gamma[k,r]*sigmaX[r,inta+k-2]*Pi[1,r]*E[:,1]
        sumJ2a=sum(sigmaJ2)/sumX[inta+k-2]
        remainderSJ2[:,inta+k-1]=sumJ2a.transpose()
        
        for r in range(0,state):
            sumJ3[:,r]= Gamma[k,0] *sigmaJ3[0,r]*Pi[:,0]+Gamma[k,1] *sigmaJ3[1,r]*Pi[:,1] + Gamma[k,2]*sigmaJ3[2,r]*Pi[:,2] + Gamma[k,3]*sigmaJ3[3,r]*Pi[:,3] 
            sigmaJ3[:,r]=sumJ3[:,r]+Gamma[k,r]*sigmaX[r,inta+k-2]*Pi[2,r]*E[:,2]
        sumJ3a=sum(sigmaJ3)/sumX[inta+k-2]
        remainderSJ3[:,inta+k-1]=sumJ3a.transpose()
        
        for r in range(0,state):
            sumJ4[:,r]= Gamma[k,0]*sigmaJ4[0,r]*Pi[:,0]+ Gamma[k,1] *sigmaJ4[1,r]*Pi[:,1] + Gamma[k,2]*sigmaJ4[2,r]*Pi[:,2] + Gamma[k,3]*sigmaJ4[3,r]*Pi[:,3] 
            sigmaJ4[:,r]= sumJ4[:,r]+ Gamma[k,r]*sigmaX[r,inta+k-2]*Pi[3,r]*E[:,3]
        sumJ4a=sum(sigmaJ4)/sumX[inta+k-2]
        remainderSJ4[:,inta+k-1]=sumJ4a.transpose()
        
        for r in range(0,state):
            sumO[:,r]= Gamma[k,0] *sigmaO[0,r]*Pi[:,0]+Gamma[k,1] *sigmaO[1,r]*Pi[:,1] + Gamma[k,2]*sigmaO[2,r]*Pi[:,2]+ Gamma[k,3]*sigmaO[3,r]*Pi[:,3]
            sigmaO[:,r]= sumO[:,r] + Gamma[k,r]*sigmaX[r,inta+k-2]*Pi[:,r]
        sumOa= sum(sigmaO)/sumX[inta+k-2]
        remainderSO[:,inta+k-1]=sumOa.transpose()
        
        for r in range(0,state):
            sumT1[:,r]= Gamma[k,0] *sigmaT1[0,r]*Pi[:,0]+Gamma[k,1] *sigmaT1[1,r]*Pi[:,1] + Gamma[k,2]*sigmaT1[2,r]*Pi[:,2] + Gamma[k,3]*sigmaT1[3,r]*Pi[:,3]
            sigmaT1[:,r]= sumT1[:,r] + Gamma[k,r]*sigmaX[r,inta+k-2]*IR[k]*Pi[:,r]
        sumT1a= sum(sigmaT1)/sumX[inta+k-2]
        remainderST1[:,inta+k-1]=sumT1a.transpose()
        
        for r in range(0,state):
            sumT2[:,r]= Gamma[k,0] *sigmaT2[0,r]*Pi[:,0]+Gamma[k,1] *sigmaT2[1,r]*Pi[:,1] + Gamma[k,2]*sigmaT2[2,r]*Pi[:,2] + Gamma[k,3]*sigmaT2[3,r]*Pi[:,3]
            sigmaT2[:,r]= sumT2[:,r] + Gamma[k,r]*sigmaX[r,inta+k-2]*IR[k]**2 *Pi[:,r]
        sumT2a= sum(sigmaT2)/sumX[inta+k-2]
        remainderST2[:,inta+k-1]=sumT2a.transpose()
        
        for r in range(0,state):
            sumT3[:,r]= Gamma[k,0] *sigmaT3[0,r]*Pi[:,0]+Gamma[k,1] *sigmaT3[1,r]*Pi[:,1] + Gamma[k,2]*sigmaT3[2,r]*Pi[:,2] + Gamma[k,3]*sigmaT3[3,r]*Pi[:,3]
            sigmaT3[:,r]= sumT3[:,r] + Gamma[k,r]*sigmaX[r,inta+k-2]*intrate[inta+k-2]*IR[k]*Pi[:,r]
        sumT3a= sum(sigmaT3)/sumX[inta+k-2]
        remainderST3[:,inta+k-1]=sumT3a.transpose()
    
    #for i in range(0,state):
        beta[i]= gamma[i]/(1-alpha[i])
        a[i]= - np.log(alpha[i])/(u+1-u) # aplha must (-) removed
        theta[i]= nu[i]* (np.sqrt(np.abs(2*a[i])))/ np.sqrt(np.abs((1-np.exp(-2*a[i]*(u+1-u)))))
    for k in range(0,batch):
        Xk1= np.matmul(Pi,sigmaX[:,inta+k-2])#Pi*sigmaX[:,inta+k-1] #
        alphaIP1 = 0
        gammaIP1=0
        nuIP1=0
        for i in range(0,state):
            a11= alpha[i]*Xk1[i]
            alphaIP1= alphaIP1 + a11
            b11=gamma[i]*Xk1[i]
            gammaIP1=gammaIP1 + b11
            c11 = nu[i]*Xk1[i]
            nuIP1 = nuIP1 + c11
        Xk2= sigmaX[:,inta+k-2]
        alphaIP2 = 0
        gammaIP2=0
        nuIP2=0
        for i in range(0,state):
            a21= alpha[i]*Xk2[i]
            alphaIP2= alphaIP2 + a21
            b21=gamma[i]*Xk2[i]
            gammaIP2=gammaIP2 + b21
            c21 = nu[i]*Xk2[i]
            nuIP2 = nuIP2 + c21
        intr[k+1] = alphaIP1*intrate[inta+k-2] + gammaIP1 # dimension issues
        Loglikelihood[k] = -0.5* np.log(2*math.pi*nuIP1**2)- (intrate[inta+k-1]- alphaIP1*intrate[inta+k-2] - gammaIP1)**2/(2*nuIP1**2)# if remove k , it works
    
    
    LogLike[:,u]= Loglikelihood[0:batch].transpose()
    F[:,u]= intr[1:batch+1] # needs editing
    x1 = sigmaX[0,inta+batch-2]  # USE ONE BEFORE LAST ELEMENT. LAST ELEMENT IS SUM/SUM (CHECK ABOVE)
    x2 = sigmaX[1,inta+batch-2]
    x3 = sigmaX[2,inta+batch-2]
    x4 = sigmaX[3,inta+batch-2]
    AL[u,:]= alpha
    GA[u,:]= gamma
    NU[u,:]= nu
    FISH_AL[u,:]= fish_alpha
    FISH_GA[u,:]= fish_gamma
    FISH_NU[u,:]= fish_nu
    PROB1[u,:] = Pi[0,0]
    PROB2[u,:] = Pi[0,1]
    PROB3[u,:] = Pi[0,2]
    PROB4[u,:] = Pi[0,3]
    PROB5[u,:] = Pi[1,0]
    PROB6[u,:] = Pi[1,1]
    PROB7[u,:] = Pi[1,2]
    PROB8[u,:] = Pi[1,3]
    PROB9[u,:] = Pi[2,0]
    PROB10[u,:] = Pi[2,1]
    PROB11[u,:] = Pi[2,2]
    PROB12[u,:] = Pi[2,3]
    PROB13[u,:] = Pi[3,0]
    PROB14[u,:] = Pi[3,1]
    PROB15[u,:] = Pi[3,2]
    PROB16[u,:] = Pi[3,3]
    
    for r in range(0,state):
        Pi[0,r]= remainderSJ1[r,inta+batch-2]/remainderSO[r,inta+batch-2]
        Pi[1,r]= remainderSJ2[r,inta+batch-2]/remainderSO[r,inta+batch-2]
        Pi[2,r]= remainderSJ3[r,inta+batch-2]/remainderSO[r,inta+batch-2]
        Pi[3,r]= remainderSJ4[r,inta+batch-2]/remainderSO[r,inta+batch-2]
        Pi=Pi.transpose() # chnage if time comes
    for r in range(0,state):
        alpha[r]= (remainderST3[r,inta+batch-2]-remainderST1[r,inta+batch-2]*gamma[r])/remainderST2[r,inta+batch-2]
    for r in range(0,state):
        gamma[r] =(remainderST1[r,inta+batch-1] - remainderST1[r,inta+batch-2]*alpha[r])/remainderSO[r,inta+batch-2]
    #gamma[2]=abs((remainderST1[2,inta+batch-1] - remainderST1[2,inta+batch-2]*alpha[2])/remainderSO[2,inta+batch-2]) 
    for r in range(0,state):
        nu[r] = math.sqrt(abs(remainderST2[r,inta+batch-1]+ (alpha[r]**2) * remainderST2[r,inta+batch-2] + (gamma[r]**2) * remainderSO[r,inta+batch-2] - 2*alpha[r]*remainderST3[r,inta+batch-2]- 2*gamma[r]*remainderST1[r,inta+batch-2] - 2*alpha[r]*gamma[r]* remainderST1[r,inta+batch-2])/remainderSO[r,inta+batch-2])
    for r in range(0,state):
        fish_alpha[r]= remainderST2[r,inta+batch-2]/nu[r]**2
    for r in range(0,state):
        fish_gamma[r] = remainderSO[r,inta+batch-2]/(2*nu[r]**2)
    for r in range(0,state):
        fish_nu[r] = (remainderSO[r,inta+batch-2]/nu[r]**2) + 3*(remainderST2[r,inta+batch-1]+ (alpha[r]**2) * remainderST2[r,inta+batch-2] + (gamma[r]**2) * remainderSO[r,inta+batch-2] - 2*alpha[r]*remainderST3[r,inta+batch-2] - 2*gamma[r]*remainderST1[r,inta+batch-1] - 2*alpha[r]*gamma[r]* remainderST1[r,inta+batch-2])/nu[r]**4
    inta= inta+batch
    inte= inte+batch 
     
LogLikeModel = LogLike.flatten('F')
summeLog = sum(LogLike)

for i in range(0,no_passes):
    LogLikelihoodTotal[i]=summeLog[i]
    AIC3st[i]= -2*LogLikelihoodTotal[i]+ 2*28
    BIC3st[i]= -2*LogLikelihoodTotal[i]+ 28*math.log(batch)

PROB = [PROB1, PROB2, PROB3, PROB4, PROB5, PROB6, PROB7, PROB8, PROB9,PROB10, PROB11, PROB12, PROB13, PROB14, PROB15, PROB16]
F42 = F.flatten('F')
m=len(F42)
forecast=F42[0:m]
K=np.linspace(1,m,960)
observation=intrate[1:m+1]


diffobservation = np.zeros(m-1)
diffforecast = np.zeros(m-1)
diffresidual = np.zeros(m-1)
for f in range(0,m-1): #=1: (m-1)
    diffobservation[f]=observation[f+1]-observation[f]
    diffforecast[f]=forecast[f+1]-forecast[f]
diffresidual=diffobservation-diffforecast;

aa=forecast
N=len(aa)
bb=observation
cc=intrate[0:m]
diff = np.zeros(N)
for i in range(0,N):
    diff[i]=(bb[i]-aa[i])**2;
MSE=(1/(N-1))*sum(diff)

N=len(aa)
absdiff1 = np.zeros(N)
absdiff2 = np.zeros(N)
RAE = np.zeros(N)
for i in range(0,N):
    absdiff1[i]=abs(aa[i]-bb[i])
    absdiff2[i]=abs(cc[i]-bb[i])
    RAE[i]=absdiff1[i]/absdiff2[i]
SRAE=np.sort(RAE)
SRAE= SRAE[SRAE!=np.inf]
s1= (N/2) -1
s2= (N/2)
MdRAE=(SRAE[479]+SRAE[480])/2
cumRAE=sum(absdiff1)/sum(absdiff2) 
cumRAE
diff1 = np.zeros(N)
APE = np.zeros(N)
for i in range(0,N):
    diff1[i]=aa[i]-bb[i]
    APE[i]=abs(diff1[i]/bb[i])
SAPE=np.sort(APE)
MdAPE=(SAPE[479]+SAPE[480])/2


print(MSE)
print(MdAPE)
print(MdRAE)
print(sum(summeLog))
print(sum(AIC3st))
print(sum(BIC3st))

