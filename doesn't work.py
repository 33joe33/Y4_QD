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

gamma = np.array([0.001,1])

J=1
d=0


def create_operator(operators):
    # create operator for multi particle states
    New_operator = 1
    for operator in operators:
        New_operator = np.kron(New_operator, operator)
    return New_operator

class system:

    def __init__(self, gamma):
        hbar = 1
        B0 = 1
        k_b = 1
        T = 1
        beta = (hbar * gamma * B0) / (k_b * T)
        w0 = gamma * B0

        w_mw = w0[0] - w0[1]

        I1,I2,I = [0,0,0,0],[0,0,0,0],[0,0,0,0]
        for n in range(4):
            I1[n] = create_operator([sigma[n],sigma[0]])
            I2[n] = create_operator([sigma[0],sigma[n]])
            I[n] = I1[n] + I2[n]

        #write d.operator and hamiltonian in stationary basis
        self.Density_Operator = 0.25*(I[0] + beta[0]*I1[3]+beta[1]*I2[3])
        
        self.Hamiltonian0 = w0[0]*I1[3] - w0[1]*I2[3]
        self.HamiltonianA = 2*(np.pi*J+d)*create_operator([sigma[3],sigma[3]])
        self.HamiltonianB = 0.5*(2*np.pi*J-d)*(create_operator([sigma_p,sigma_m])+create_operator([sigma_m,sigma_p]))
        self.HamiltonianMW = w_mw*(create_operator([sigma[0],sigma[1]]))
        
        self.Hamiltonian = self.Hamiltonian0 + self.HamiltonianA + self.HamiltonianB
        
        #diagonalise H and rewrite d.operator in new basis
        eigval,eigvec = linalg.eig(self.Hamiltonian)
        idx = eigval.argsort()[::-1]
        self.eigval = eigval[idx]
        self.eigvec = eigvec[:,idx]
        self.eigvec_i = linalg.inv(self.eigvec)

        self.H = self.diag(self.Hamiltonian) + self.diag(self.HamiltonianMW)
        
        self.Density_Operator = self.diag(self.Density_Operator)

        
    def evolve(self):
        N = 1000
        End = 10
        T = np.linspace(0, End, N)
        delta_t = End/N
        
        P1,P2,P3,P4 = np.zeros(T.size),np.zeros(T.size),np.zeros(T.size),np.zeros(T.size)

        M1z,M2z = np.zeros(T.size),np.zeros(T.size)
        z1,z2 = np.zeros(T.size),np.zeros(T.size)

        I1z = self.diag(create_operator([sigma[3],sigma[0]]))
        I2z = self.diag(create_operator([sigma[0],sigma[3]]))


        for n in range(N):
            self.Density_Operator = linalg.expm(1.j * self.H * delta_t) @ (self.Density_Operator @ linalg.expm(-1.j * self.H * delta_t))
            operator = self.undiag(self.Density_Operator)

            #extract populations
            P1[n] = operator[0,0] 
            P2[n] = operator[1,1]
            P3[n] = operator[2,2]
            P4[n] = operator[3,3]
            
            M1z[n] = P1[n] - P2[n]
            M2z[n] = -(P3[n] - P4[n])
            
            z1[n] = np.trace(I1z @ self.Density_Operator)
            z2[n] = np.trace(I2z @ self.Density_Operator)
        
        plt.figure()
        plt.plot(T,P1,'k',T,P2,'r',T,P3,'b',T,P4,'g')
        plt.xlabel('time')
        plt.ylabel('Populations')
        


        plt.figure()
        plt.plot(T,M1z,'k',T,M2z,'r')
        
        plt.figure()
        plt.plot(T,z1,'k',T,z2,'r')
        
        
    
    def diag(self, operator):
        return (self.eigvec_i @ (operator @ self.eigvec))
    
    def undiag(self, operator):
        return (self.eigvec @ (operator @ self.eigvec_i))

x = system(gamma)
a=x.Hamiltonian
x.evolve()


