import numpy as np
import matplotlib.pyplot as plt
import math

def ridge_regression(tX, tY, l=0.15):
  n, dim = tX.shape
  diagonal = np.array([n*l]*dim)
  A = np.diag(diagonal)+ np.dot(tX.T, tX)
  b = np.dot(tX.T, tY)
  theta = np.dot(np.linalg.inv(A), b)
  return theta


#first 10 validation set, last 40 training set
x = np.genfromtxt('hw1_ridge_x.dat', delimiter=',', dtype = float)
y = np.genfromtxt('hw1_ridge_y.dat', dtype = float).reshape(50,1)
tX = x[-40:]
tY = y[-40:]
vX = x[0:10]
vY = y[0:10]

l = 0.15
theta = ridge_regression(tX, tY, l)
print("theta:",theta)

import matplotlib.pyplot as plt 
tn = tX.shape[0]
vn = vX.shape[0]
tloss = []
vloss = []
index = -np.arange(0,5,0.1)
for i in index:
    w = ridge_regression(tX,tY,10**i)
    tloss = tloss + [np.sum((np.dot(tX,w)-tY)**2)/tn/2]
    vloss = vloss + [np.sum((np.dot(vX,w)-vY)**2)/vn/2]

# print(tloss, vloss)
log_vloss = np.log(vloss)
y_min = np.amin(log_vloss)
number = np.where(log_vloss == y_min)
xpos = index[number]
print("lowestpoint:",xpos, y_min)

t = plt.plot(index,np.log(tloss), 'r', label = 'log(tloss')
v = plt.plot(index,np.log(vloss), 'b', label = 'log(vloss)')
plt.legend()
plt.show()
