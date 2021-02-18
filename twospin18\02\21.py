import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


sigma_0 = np.array([[1,0],[0,1]])
sigma_1 = np.array([[0, 1], [1, 0]])
sigma_2 = np.array([[0, -1.j], [1.j, 0]])
sigma_3 = np.array([[1, 0], [0, -1]])
sigma = [sigma_0,0.5*sigma_1,0.5*sigma_2,0.5*sigma_3]
sigma_p = sigma[1] + 1.j*sigma[2]
sigma_m = sigma[1] - 1.j*sigma[2]

gamma = np.array([1,2])
w1,w2 = 1,1
J=5
d=1



def create_operator(operators):
    # create operator for multi particle states
    New_operator = 1
    for operator in operators:
        New_operator = np.kron(New_operator, operator)
    return New_operator

class system:

    # set pauli matrices
    def __init__(self, gamma):
        hbar = 1
        B0 = 1
        k_b = 1
        T = 1
        beta = (hbar * gamma * B0) / (k_b * T)

        I1,I2,I = [0,0,0,0],[0,0,0,0],[0,0,0,0]
        for n in range(4):
            I1[n] = create_operator([sigma[n],sigma[0]])
            I2[n] = create_operator([sigma[0],sigma[n]])
            I[n] = I1[n] + I2[n]

        self.Density_Operator = 0.125*(I[0] + beta[0]*I1[3]+beta[1]*I2[3])
        print(self.Density_Operator)
        
        self.Hamiltonian0 = w1*I1[3] + w2*I2[3]
        self.HamiltonianA = 2*(np.pi*J+d)*create_operator([sigma[3],sigma[3]])
        self.HamiltonianB = 0.5*(2*np.pi*J-d)*(create_operator([sigma_p,sigma_m])+create_operator([sigma_m,sigma_p]))
        self.Hamiltonian = self.Hamiltonian0 + self.HamiltonianA + self.HamiltonianB

        #print(self.Hamiltonian)
        
    def evolve(self,operator):
        N = 1000
        End = 1*np.pi
        T = np.linspace(0, End, N)
        delta_t = End/N
        P1,P2,P3,P4 = np.zeros(T.size),np.zeros(T.size),np.zeros(T.size),np.zeros(T.size)
        C1,C2 = np.zeros(T.size),np.zeros(T.size)
        for n in range(N):
            operator = np.dot(expm(1.j * self.Hamiltonian * delta_t), np.dot(operator,expm(-1.j * self.Hamiltonian * delta_t)))
            #extract populations and coherences
            P1[n] = operator[0,0] 
            P2[n] = operator[1,1]
            P3[n] = operator[2,2]
            P4[n] = operator[3,3]
            C1[n] = operator[1,2]
            C2[n] = operator[2,1]
            
        Mz1 = P1-P2
        Mz2 = P3-P4
        plt.figure()
        plt.plot(T,P1,'k',T,P2,'r',T,P3,'b',T,P4,'g')
        plt.figure()
        plt.plot(T,C1,'k',T,C2,'r')
        plt.figure()
        plt.plot(T,Mz1,'k',T,Mz2,'r')


#TODO generalise for n particles
x = system(gamma)

x.evolve(x.Density_Operator)


# print(x.Density_Operator)
# print(x.expectation_value(x.sigma[2]))
# x.Time_evolution(x.sigma[1]
#TODO: couple spins and get 2 spin system increase dimension  electron + proton different magnet constant 
