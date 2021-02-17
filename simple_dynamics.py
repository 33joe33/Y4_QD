import matplotlib
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from itertools import repeat


sigma_0 = np.array([[1,0],[0,1]])
sigma_1 = np.array([[0, 1], [1, 0]])
sigma_2 = np.array([[0, -1.j], [1.j, 0]])
sigma_3 = np.array([[1, 0], [0, -1]])

gamma = [1,2]
B = 1
k_b = 1
T = 1
beta = 1 / (k_b * T)
J=1
d=1


def create_operator(operators):
    # create operator for multi particle states
    New_operator = 1
    for operator in operators:
        New_operator = np.kron(New_operator, operator)
    return New_operator

class system:


    def Com(self, A, B):
        return np.dot(A, B) - np.dot(B, A)


    sigma = [0.5*np.array([[0, 1], [1, 0]]), 0.5*np.array([[0, -1.j], [1.j, 0]]), 0.5*np.array([[1, 0], [0, -1]]),np.array([[1,0],[0,1]])]

    # set pauli matrices
    def __init__(self, Hamiltonian=B * create_operator([sigma[2]]),No_particles=1):
        self.Hamiltonian = Hamiltonian
        # self.Density_Operator = np.array(0.5*np.identity(2) +0.5*B*self.sigma[2])
        self.Density_Operator = np.array(expm(beta * Hamiltonian))
        self.Density_Operator = self.Density_Operator / np.sum(self.Density_Operator)

        self.sigma  =[create_operator([item]+[self.sigma[3] for r in range(No_particles-1) ]) for item in self.sigma]+[create_operator([self.sigma[3] for r in range(No_particles-1)]+[item]) for item in self.sigma]

    def rotation(self, theta):
        # TODO: work out how to rotate
        # general rotation about all axis x=0, y=1 and z =2
        R = np.array([[np.cos(theta[2]) * np.cos(theta[1]),
                       np.cos(theta[2]) * np.sin(theta[1]) * np.sin(theta[0]) - np.sin(theta[2]) * np.cos(theta[0]),
                       np.cos(theta[2]) * np.sin(theta[1]) * np.cos(theta[0]) - np.sin(theta[2]) * np.sin(theta[0])],

                      [np.sin(theta[2]) * np.cos(theta[1]),
                       np.sin(theta[2]) * np.sin(theta[1]) * np.sin(theta[0]) + np.cos(theta[2]) * np.cos(theta[0]),
                       np.sin(theta[2]) * np.sin(theta[1]) * np.cos(theta[0]) - np.cos(theta[2]) * np.sin(theta[0])],

                      [-np.sin(theta[1]), np.cos(theta[1]) * np.sin(theta[0]), np.cos(theta[1]) * np.cos(theta[0])]])
        return R



    def rotating_frame(self,phi):
        #[[np.cos()],[]]

        # general rotation about all axis x=0, y=1 and z =2
        R_p =expm(-1.j *(phi[0]*self.sigma[0]+phi[1]*self.sigma[1]+phi[2]*self.sigma[2]))
        R_m =expm(1.j *(phi[0]*self.sigma[0]+phi[1]*self.sigma[1]+phi[2]*self.sigma[2]))

        print(self.Hamiltonian)
        self.Hamiltonian=np.dot(R_p, np.dot(self.Hamiltonian,R_m))
        print(self.Hamiltonian)



    def Time_evolution(self, Operator):
        temp_op = Operator
        time_evolution_list= []
        # TODO: append a operator thing
        delta_t = 0.01
        T = np.arange(0, np.pi, delta_t)
        for t in T:
            temp_op = np.dot(expm(1.j * self.Hamiltonian * delta_t), np.dot(temp_op,expm(-1.j * self.Hamiltonian * delta_t)))

            time_evolution_list.append(self.expectation_value(temp_op))

        plt.plot(T,time_evolution_list)


    def expectation_value(self, Operator):
        return (np.trace(np.dot(Operator, self.Density_Operator)))
    def Change_Hamiltonian(self, New_Hamiltonian):
        self.Hamiltonian=New_Hamiltonian
    def Produce_Graph(self):
        x.Time_evolution(x.sigma[0])
        x.Time_evolution(x.sigma[1])
        x.Time_evolution(x.sigma[2])
        plt.title("precession in magnetic field along z axis")
        plt.legend(["I_x ", "I_y ", "I_z"])
        plt.show()

Hamiltonian  = B*(gamma[0]*create_operator([sigma_3,sigma_0])+ gamma[1]*create_operator([sigma_0,sigma_3]))+2*(np.pi*J+d)*create_operator([sigma_3,sigma_3]) +0.5*(2*np.pi*J-d)*(create_operator([sigma_1+1.j*sigma_2,sigma_1+1.j*sigma_2])+create_operator([sigma_1-1.j*sigma_2,sigma_1-1.j*sigma_2]))
#TODO generalise for n particles
print(Hamiltonian)
x = system(Hamiltonian,2)
x.Produce_Graph()

x.rotating_frame([np.pi/2,0,0])
x.Produce_Graph()



# print(x.Density_Operator)
# print(x.expectation_value(x.sigma[2]))
# x.Time_evolution(x.sigma[1]
#TODO: couple spins and get 2 spin system increase dimension  electron + proton different magnet constant 