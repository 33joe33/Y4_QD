import matplotlib
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt


sigma_1 = np.array([[0, 1], [1, 0]])
sigma_2 = np.array([[0, -1.j], [1.j, 0]])
sigma_3 = np.array([[1, 0], [0, -1]])

gamma = 1
B = 1
k_b = 1
T = 1
beta = 1 / (k_b * T)


def create_operator(operators):
    # create operator for multi particle states
    New_operator = 1
    for operator in operators:
        New_operator = np.kron(New_operator, operator)
    return New_operator

class system:


    def Com(self, A, B):
        return np.dot(A, B) - np.dot(B, A)

    sigma = [np.array([[0, 1], [1, 0]]), np.array([[0, -1.j], [1.j, 0]]), np.array([[1, 0], [0, -1]])]

    # set pauli matrices
    def __init__(self, Hamiltonian=B * create_operator([sigma[2]])):
        self.Hamiltonian = Hamiltonian
        # self.Density_Operator = np.array(0.5*np.identity(2) +0.5*B*self.sigma[2])
        self.Density_Operator = np.array(expm(beta * Hamiltonian))
        self.Density_Operator = self.Density_Operator / np.sum(self.Density_Operator)

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

    def rotating_frame(self):
        pass

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

x = system()

x.Time_evolution(x.sigma[2])

x.Change_Hamiltonian(B * create_operator([x.sigma[0]]))
x.Time_evolution(x.sigma[2])
x.Time_evolution(x.sigma[1])
plt.title("precession in magnetic field along x axis")
plt.legend(["I_z ","I_z' ","I_y'"])
plt.show()
# print(x.Density_Operator)
# print(x.expectation_value(x.sigma[2]))
# x.Time_evolution(x.sigma[1]
