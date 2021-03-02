import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
#rrr
w_n, w_e = 600e6, 395e9
J = 1
d = 0


class create_Operators:

    def __init__(self, dimension):
        self.sigma_0 = [[1, 0], [0, 1]]
        self.sigma_1 = [[0, 1], [1, 0]]
        self.sigma_2 = [[0, -1.j], [1.j, 0]]
        self.sigma_3 = [[1, 0], [0, -1]]

        self.pauliM = [self.sigma_0, self.sigma_1, self.sigma_2, self.sigma_3]
        self.dimension = dimension

        self.operators = self.create()

    def create(self):
        temp_matrix = 1
        for i in range(self.dimension):
            temp_matrix = np.kron(temp_matrix, self.pauliM)

        return temp_matrix

    def get(self, select):
        index = 0
        multiplier = 1
        for dim in select:
            index += dim * multiplier
            multiplier *= 4
        return self.operators[index]


class system:
    def __init__(self, w):
        self.Pauli_Matrices = create_Operators(len(w))

        hbar = 1.054571817e-34
        k_b = 1.38e-23
        T = 300
        beta = (hbar * w) / (k_b * T)
        a = 1
        w_mw = a * (w[0] - w[1])
        
        self.Hamiltonian =  - w[0] * self.Pauli_Matrices.get([3, 0]) + w[1] * self.Pauli_Matrices.get([0, 3])
        - w_mw * self.Pauli_Matrices.get([3,0]) + 2 * np.pi * J * self.Pauli_Matrices.get([0,3]) @ self.Pauli_Matrices.get([3,0])
        
        self.HamiltonianMW = w_mw * self.Pauli_Matrices.get([1,0])
        
        self.Hamiltonian = self.Hamiltonian + self.HamiltonianMW

        self.Density_Operator = self.Pauli_Matrices.get([0, 0]) + beta[0] * self.Pauli_Matrices.get([3, 0]) + beta[
            1] * self.Pauli_Matrices.get([0, 3])
        
        #print(self.Density_Operator)

    def evolve(self):
        #are we evolving the density operator
        N = 1000
        End = 1
        T = np.linspace(0, End, N)
        m1, m2 = np.zeros([4, T.size]), np.zeros([4, T.size])
        #print(self.Hamiltonian)
        delta_t = End / N
        
        for n in range(N):
            self.Density_Operator = linalg.expm(1.j *self.Hamiltonian*delta_t) @ self.Density_Operator @ linalg.expm(-1.j *self.Hamiltonian*delta_t)
            
            for a in[1,2,3]:
                m1[a,n] =np.trace(self.Pauli_Matrices.get([a,0])@ self.Density_Operator)
                m2[a,n] =np.trace(self.Pauli_Matrices.get([0,a])@ self.Density_Operator)
                
            m1[0,n] = ((m1[1,n]**2)+(m1[2,n]**2)+(m1[3,n]**2))**0.5
            m2[0,n] = ((m2[1,n]**2)+(m2[2,n]**2)+(m2[3,n]**2))**0.5
                
        #print(self.Density_Operator)
        fign, ((axn1, axn2), (axn3, axn4)) = plt.subplots(2, 2)
        fige, ((axe1, axe2), (axe3, axe4)) = plt.subplots(2, 2)



        axe1.plot(T, m1[1, :], 'k')
        axe1.set(xlabel='t', ylabel='Mxe')

        axn1.plot( T, m2[1, :], 'r')
        axn1.set(xlabel='t', ylabel='Mxn')

        axe2.plot(T, m1[2, :], 'k')
        axe2.set(xlabel='t', ylabel='Mye')

        axn2.plot(T, m2[2, :], 'r')
        axn2.set(xlabel='t', ylabel='Myn')

        axe3.plot(T, m1[3, :], 'k')
        axe3.set(xlabel='t', ylabel='Mze')

        axn3.plot(T, m2[3, :], 'r')
        axn3.set(xlabel='t', ylabel='Mzn')
        
        axe4.plot(T, m1[0, :], 'k')
        axe4.set(xlabel='t', ylabel='Me')
        
        axn4.plot(T, m2[0, :], 'r')
        axn4.set(xlabel='t', ylabel='Mn')


        plt.show()





#x=system(np.array([1,1]))
x = system(np.array([w_e,w_n]))
x.evolve()