#!/usr/bin/env python
# coding: utf-8

# In[125]:


import numpy as np
import matplotlib.pyplot as plt

class pm:
  S = 50 # State space  Cardinality
  A = 10 # Action space cardinality
  y = 0.9 # Discount factor
  # Uncertianty radius
  a_p = 0.1
  a_sa = 0.1*np.ones((S,A))
  a_s = 0.1*np.ones(S)
  N = 1000
  eta = 0.01 # learning rate
  # Reward function
  R = np.random.randn(S,A)
  # Transition Kernel
  P = np.random.rand(S,A,S)
  for a in range(A):
    for s in range(S):
      P[s,a] = P[s,a]/np.sum(P[s,a])



# Random Initial policy
pi = np.random.rand(pm.S,pm.A)
for s in range(pm.S):
    pi[s] = pi[s]/np.sum(pi[s])

Q = np.random.randn(pm.S,pm.A)



# In[139]:


# Occupation measure Calculation
def Occupation(pi):
  Ppi = np.sum(pm.P*np.reshape(np.repeat(pi,pm.S),(pm.S,pm.A,pm.S)),axis=1)
  temp = np.mean(Ppi,axis=0)
  d = temp
  for i in range(1000):
    temp = pm.y*temp@Ppi
    d = d + temp
  return d

# Non Robust Q learning
def Qtrain(pi,Q):
  for n in range(10000):
    Q = pm.R + pm.y*pm.P@(np.sum(Q*pi,axis=1))
  return Q

# sa rect Lp Robust Q learning
def Qsatrain(pi,Q):
  for n in range(10000):
    Q = pm.R -pm.a_sa +  pm.y*pm.P@(np.sum(Q*pi,axis=1))
  return Q


# s rect Lp Robust Q learning
def Qsptrain(pi,Q,p):
  q = 1/(1-1/p) # Holders conjugate
  reg = (np.diag(1/((np.sum(pi**q,axis=1))**(1/q)))@pi)**(q-1)
  for n in range(10000):
    Q = pm.R -np.reshape(np.repeat(pm.a_s,pm.A),(pm.S,pm.A))*reg+  pm.y*pm.P@(np.sum(Q*pi,axis=1))
  return Q


#  Lp Robust Q learning
def Qptrain(pi,Q,p):
  # Holders conjugate
  q = 1/(1-1/p)
  d = Occupation(pi)
  # Regularizer calculation
  D = d@pi
  reg = (D/((np.sum(D**q))**(1/q)))**(q-1)

  # Q-learning
  for n in range(10000):
    Q = pm.R -pm.a_p*reg + pm.y*pm.P@(np.sum(Q*pi,axis=1))
  return Q



# Projection Operator:  https://gist.github.com/mblondel/c99e575a5207c76a99d714e8c6e08e89
def projection_simplex(V, z=1, axis=None):
    """
/    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0)

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1).T

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()

def policyUpdate(pi,Q):
  d = Occupation(pi)
  D = np.reshape(np.repeat(d,pm.A),(pm.S,pm.A))
  return projection_simplex(pi + pm.eta*D*Q,axis=1)

def RobustReturn(pi,mode='nr',p=2):
    q = 1/(1-1/p)
    d = Occupation(pi)
#     print(q)
    nr = np.sum(d*np.sum(pm.R*pi,axis=1)) # non robust return
    if mode=='nr': # non robust 
        return nr
    if mode == 'sa': # sa-rectangular
        return nr - np.sum((np.diag(d)@pi)*pm.a_sa)
    
    if mode == 's': # s-rectangular
        return  nr - np.sum(d*pm.a_s*((np.sum(pi**q,axis=1))**(1/q))) # robust return
     
    if mode == 'p': # non-rectangular
        return nr - pm.a_p*((np.sum((np.diag(d)@pi)**q))**(1/q)) # robust return
    
    

def train(pi,Q,mode='nr',p=2):
  Return = []
  q = 1/(1-1/p)
  for i in range(pm.N):
    d = Occupation(pi)
    if mode=='nr': # non robust
      Q = Qtrain(pi,Q)
    if mode == 'sa': # sa-rectangular
      Q = Qsatrain(pi,Q)
    if mode == 's': # s-rectangular
      Q = Qsptrain(pi,Q,p=p)
    if mode == 'p': # non-rectangular
      Q = Qptrain(pi,Q,p=p)
    r = RobustReturn(pi,mode=mode,p=p)
    Return.append(r)
    pi = policyUpdate(pi,Q)
  return Return

RR = []
for mode in ['nr','sa']:
  print(mode)  
  Q = np.random.randn(pm.S,pm.A)
  Rl = train(pi,Q,mode=mode)
  RR.append(Rl)
legend = ['non-robust', 'sa']


# S-rect robust
mode ='s'
for p in [2,3,5,7,10,20,50,100]:
  print(mode,p)  
  Q = np.random.randn(pm.S,pm.A)
  Rl = train(pi,Q,mode=mode,p=p)
  legend.append( 's_L{}'.format(p))
  RR.append(Rl)


# Lp robust  
mode ='p'
for p in [2,3,5,7,10,20,50,100]:
  print(mode,p)  
  Q = np.random.randn(pm.S,pm.A)
  Rl = train(pi,Q,mode=mode,p=p)
  legend.append( 'L{}'.format(p))
  RR.append(Rl)
  
# Saving Plotting
np.savetxt('RGP_conv_icml24_differentPs_S{}A{}.txt'.format(pm.S,pm.A),RR)
plt.plot(np.transpose(RR[0]),'.',color='blue')
plt.plot(np.transpose(RR[1]),'.',color='black')
plt.plot(np.transpose(RR[2:10]),'--')
plt.plot(np.transpose(RR[10:]))
plt.legend(legend)
plt.title('Convergence of Robust Policy Gradient')
plt.xlabel('Iteration')
plt.ylabel('Robust Return')
plt.savefig('RGP_conv_icml24_differentPs_S{}A{}N{}'.format(pm.S,pm.A,pm.N))
plt.clf()

# sp 
plt.plot(np.transpose(RR[0]),'--',color='blue')
plt.plot(np.transpose(RR[1]),'.',color='black')
plt.plot(np.transpose(RR[2:10]))
plt.title('Convergence of s-rect Lp Robust Policy Gradient')
plt.xlabel('Iteration')
plt.ylabel('Robust Return')
plt.legend(['nr','sa','s_L2','s_L3','s_L5','s_L7','s_L10','s_L20','s_L50','s_L100'])
plt.savefig('RGP_conv_icml24_differentPs_s_Lp_S{}A{}N{}'.format(pm.S,pm.A,pm.N))
plt.clf()
# lp 
plt.plot(np.transpose(RR[0]),'--',color='blue')
plt.plot(np.transpose(RR[1]),'.',color='black')
plt.plot(np.transpose(RR[10:]))
plt.title('Convergence of  Lp Robust Policy Gradient')
plt.xlabel('Iteration')
plt.ylabel('Robust Return')
plt.legend(['nr','sa','L2','L3','L5','L7','L10','L20','L50','L100'])
plt.savefig('RGP_conv_icml24_differentPs_Lp_S{}A{}N{}'.format(pm.S,pm.A,pm.N))















