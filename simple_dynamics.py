import matplotlib
import numpy as np
import matplotlib.pyplot as plt

sigma_1 = np.array([[0,1],[1,0]])
sigma_2 = np.array([[0,-1.j],[1.j,0]])
sigma_3 = np.array([[1,0],[0,-1]])
gamma = 1
B=1
k_b =1
T=1
class densitiy_operator:
    def __init__(self, matrix):
        self.D_O = matrix


##########################################################
# 1 spin 1/2 particle

onep =densitiy_operator(np.array([[0.5, 0],[0,0.5]]))
print( '1 spin 1/2 particle polarisation'+str(np.trace(onep.D_O*sigma_3)))
############################################################
#2 state spin1/2 ensemble
ensemble=densitiy_operator(np.array([[np.exp((gamma*B)/(2*k_b*T)),0],[0,np.exp(-(gamma*B)/(2*k_b*T))]]))
print('2 state spin1/2 ensemble '+str(np.trace(ensemble.D_O*sigma_3)))
############################################################

# 2 spin 1/2 particles
sigma_3_2 =np.array([[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,-1]])
twop =densitiy_operator(np.array([[0.25,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0.25]]))
print('2 spin 1/2 particles '+str(np.trace(twop.D_O*sigma_3_2)))
###########################################################################
# do i just put time evolution in the density operator ?

