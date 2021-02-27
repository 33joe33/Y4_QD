import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt


sigma_0 = np.array([[1,0],[0,1]])
sigma_1 = np.array([[0, 1], [1, 0]])
sigma_2 = np.array([[0, -1.j], [1.j, 0]])
sigma_3 = np.array([[1, 0], [0, -1]])
sigma = [sigma_0,0.5*sigma_1,0.5*sigma_2,0.5*sigma_3]
sigma_p = sigma[1] + 1.j*sigma[2]
sigma_m = sigma[1] - 1.j*sigma[2]

class system:

    def __init__(self, w):
        hbar = 1#1.054571817e-34
        k_b = 1#1.38e-23
        T = 1#300
        beta = (hbar * w) / (k_b * T)
        a=1
        w_mw = a*(w[0] - w[1])

        I1,I2,I = [0,0,0,0],[0,0,0,0],[0,0,0,0]
        for n in range(4):
            I1[n] = np.kron(sigma[n],sigma[0])
            I2[n] = np.kron(sigma[0],sigma[n])
            I[n] = I1[n] + I2[n]

        Hamiltonian0 = w[0]*I1[3] - w[1]*I2[3]
        HamiltonianA = 2*(np.pi*J+d)*np.kron(sigma[3],sigma[3])
        HamiltonianB = 0.5*(2*np.pi*J-d)*(np.kron(sigma_p,sigma_m)+np.kron(sigma_m,sigma_p))
        HamiltonianMW = w_mw*I2[1]
        
        Hamiltonian = Hamiltonian0 + HamiltonianA + HamiltonianB
        
        #diagonalise H and rewrite d.operator in new basis
        eigval,eigvec = linalg.eig(Hamiltonian)
        idx = eigval.argsort()[::-1]
        self.eigval = eigval[idx]
        self.eigvec = eigvec[:,idx]
        self.eigvec_i = linalg.inv(self.eigvec)

        self.H = self.diag(Hamiltonian) + self.diag(HamiltonianMW)

        self.I1,self.I2 = [0,0,0,0],[0,0,0,0]
        for n in range(4):
            self.I1[n] = self.diag(I1[n])
            self.I2[n] = self.diag(I2[n])

        self.Density_Operator = 0.125*(I[0] + beta[0]*I1[3] + beta[1]*I2[3])
        print(self.Density_Operator)

        self.Density_Operator = self.diag(self.Density_Operator)

        
    def evolve(self):
        N = 1000
        End = 100
        T = np.linspace(0, End, N)
        delta_t = End/N
        
        m1,m2 = np.zeros([4,T.size]),np.zeros([4,T.size])

        for n in range(N):
            self.Density_Operator = linalg.expm(1.j * self.H * delta_t) @ (self.Density_Operator @ linalg.expm(-1.j * self.H * delta_t))

            for a in [1,2,3]:
                m1[a,n] = np.trace(self.I1[a] @ self.Density_Operator)
                m2[a,n] = np.trace(self.I2[a] @ self.Density_Operator)
            
            m1[0,n] = ((m1[1,n]**2)+(m1[2,n]**2)+(m1[3,n]**2))**0.5
            m2[0,n] = ((m2[1,n]**2)+(m2[2,n]**2)+(m2[3,n]**2))**0.5   
         
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        
        ax1.plot(T,m1[0,:],'k',T,m2[0,:],'r')
        ax1.set(xlabel='t',ylabel='M')

        ax2.plot(T,m1[1,:],'k',T,m2[1,:],'r')
        ax2.set(xlabel='t',ylabel='Mx')
        
        ax3.plot(T,m1[2,:],'k',T,m2[2,:],'r')
        ax3.set(xlabel='t',ylabel='My')
        
        ax4.plot(T,m1[3,:],'k',T,m2[3,:],'r')
        ax4.set(xlabel='t',ylabel='Mz')
        
    def diag(self, operator):
        return (self.eigvec_i @ (operator @ self.eigvec))
    
    def undiag(self, operator):
        return (self.eigvec @ (operator @ self.eigvec_i))

w_n,w_e = 600e6,395e9
w = np.array([1,10])
#2.002,2.006,2.004 - g tensor
#g=2.002
J=1
d=0

x = system(w)

x.evolve()
