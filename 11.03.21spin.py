import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

class create_Operators:
    def __init__(self, dimension):
        self.sigma_0 = np.array([[1, 0], [0, 1]])
        self.sigma_1 = np.array([[0, 1], [1, 0]])
        self.sigma_2 = np.array([[0, -1.j], [1.j, 0]])
        self.sigma_3 = np.array([[1, 0], [0, -1]])

        self.pauliM = [self.sigma_0, self.sigma_1, self.sigma_2, self.sigma_3]
        self.dimension = dimension

        self.operators = self.create()
        
        self.sigma_p = self.sigma_1 + 1.j*self.sigma_2
        self.sigma_m = self.sigma_1 - 1.j*self.sigma_2

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
        "create kronecker products"
        self.Pauli_Matrices = create_Operators(len(w))

        "Define microwave and electron offset frequencies"
        w_mw = w_e - w_n
        w_off = w_e - w_mw
        w_1 = 1e6

        "Define Hamiltonian in electron rotating frame"
        self.H_z =  w_off * self.Pauli_Matrices.get([3, 0]) - w_n * self.Pauli_Matrices.get([0, 3]) # background z field
        
        self.Hamiltonian = self.H_z

        self.HamiltonianMW = w_1 * self.Pauli_Matrices.get([1,0]) #microwave field 
        
        "Diagonalise Hamiltonian"
        self.eigval,self.eigvec = linalg.eig(self.Hamiltonian)
        self.eigvec_i = linalg.inv(self.eigvec)

        self.Hamiltonian = self.diag(self.Hamiltonian) + self.diag(self.HamiltonianMW)
        
        "Define density operator and convert to Hamiltonian basis"
        self.Density_Operator = (0.5**len(w)) * (self.Pauli_Matrices.get([3,0]) + self.Pauli_Matrices.get([0,3]))
        self.Denisty_Operator = self.diag(self.Density_Operator)
        #hbar = 1.054e-34
        #k_b = 1.38e-23
        #T = 300
        #beta = (hbar * w) / (k_b * T) 
        #self.Density_Operator = (0.5**len(w)) * (self.Pauli_Matrices.get([0, 0]) + beta[0]\
            #* self.Pauli_Matrices.get([3, 0]) + beta[1] * self.Pauli_Matrices.get([0, 3]))
            
        "Convert principle x,y,z Pauli matrices to Hamiltonian basis"
        self.S,self.I = [0,0,0,0],[0,0,0,0]
        for a in[0,1,2,3]:
            self.S[a] = self.diag(self.Pauli_Matrices.get([a,0]))
            self.I[a] = self.diag(self.Pauli_Matrices.get([0,a]))
            
            
            
    "Convert an operator to basis of diagonalised Hamiltonian"
    def diag(self, operator):
        return (self.eigvec_i @ (operator @ self.eigvec))
    
    "Return an operator to Zeeman basis"
    def undiag(self, operator):
        return (self.eigvec @ (operator @ self.eigvec_i))



    def evolve(self):
        "Define time scale"
        N = 1000
        End = 1e-8
        T = np.linspace(0, End, N)
        delta_t = End / N
        
        M_e, M_n = np.zeros([4, T.size]), np.zeros([4, T.size])

        for n in range(N):
            "Time evolve density operator"
            self.Density_Operator = linalg.expm(1.j *self.Hamiltonian*delta_t) @ self.Density_Operator @ linalg.expm(-1.j *self.Hamiltonian*delta_t)

            "Calculate x,y,z magnetisations by tracing density operator with Pauli matrices"
            for a in[1,2,3]:
                M_e[a,n] =np.trace(self.S[a] @ self.Density_Operator)
                M_n[a,n] =np.trace(self.I[a] @ self.Density_Operator)
           
        "Calculate total magnetisation"
        M_e[0,:] = ((M_e[1,:]**2)+(M_e[2,:]**2)+(M_e[3,:]**2))**0.5
        M_n[0,:] = ((M_n[1,:]**2)+(M_n[2,:]**2)+(M_n[3,:]**2))**0.5

        "Plot x,y,z and total magnetisations for nucleus and electron"
        fige, ((axe1, axe2), (axe3, axe4)) = plt.subplots(2, 2)
        fige.suptitle('Electron Magnetisation', fontsize = 15)
        fign, ((axn1, axn2), (axn3, axn4)) = plt.subplots(2, 2)
        fign.suptitle('Nuclear Magnetisation', fontsize = 15)
        
        "x magnetisation"
        axe1.plot(T, M_e[1, :], 'k')
        axe1.set(xlabel='t', ylabel='$M_x$')

        axn1.plot( T, M_n[1, :], 'r')
        axn1.set(xlabel='t', ylabel='$M_x$')

        "y magnetisation"
        axe2.plot(T, M_e[2, :], 'k')
        axe2.set(xlabel='t', ylabel='$M_y$')

        axn2.plot(T, M_n[2, :], 'r')
        axn2.set(xlabel='t', ylabel='$M_y$')

        "z magnetisation"
        axe3.plot(T, M_e[3, :], 'k')
        axe3.set(xlabel='t', ylabel='$M_z$')

        axn3.plot(T, M_n[3, :], 'r')
        axn3.set(xlabel='t', ylabel='$M_z$')
        
        "total magnetisation"
        axe4.plot(T, M_e[0, :], 'k')
        axe4.set(xlabel='t', ylabel='M')
        
        axn4.plot(T, M_n[0, :], 'r')
        axn4.set(xlabel='t', ylabel='M')



"electron and nuclear larmor frequencies"
w_n, w_e = 600e6, 395e9
"J coupling constant"
J = 0

"run system"
x = system(np.array([w_e,w_n]))
x.evolve()
