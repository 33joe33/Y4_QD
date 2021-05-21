import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

class system:
    "Create Hamiltonian and Density Operator"
    def __init__(self, w_e):
        
        "Define Pauli matrices"
        sigma_0 = np.array([[1, 0], [0, 1]])
        sigma_1 = 0.5 * np.array([[0, 1], [1, 0]])
        sigma_2 = 0.5 * np.array([[0, -1.j], [1.j, 0]])
        sigma_3 = 0.5 * np.array([[1, 0], [0, -1]])
        self.pauli = [sigma_0, sigma_1, sigma_2, sigma_3]
        
        "raising and lowering operators"
        self.S_p = self.pauli[1] + 1j*self.pauli[2]
        self.S_m = self.pauli[1] - 1j*self.pauli[2]
        
        self.S = self.pauli
        
        "Define Hamiltonian"
        self.Hamiltonian = w_e * self.S[3]
        
        "Convert Hamiltonian to Liouville space"
        self.L_H = np.kron(self.Hamiltonian, np.identity(2)) - np.kron(np.identity(2), self.Hamiltonian)
        
        "Electron thermal polarisation"
        h = 6.626e-34
        k_b = 1.38e-23
        T = 100
        beta = (h * w_e) / (k_b * T) 
        self.p_0 = np.tanh(beta/2)

        "Define initial density operator"
        #self.Density_Operator = self.pauli[0] - 2 * self.p_0 * self.pauli[3]
        self.Density_Operator = self.S[3]
        
    
    def relax(self):
        Sz_L = np.kron(self.S[3], np.identity(2))
        Sz_R = np.kron(np.identity(2), self.S[3])
        Sz_C = Sz_L - Sz_R
        
        Sp_L = np.kron(self.S_p, np.identity(2))
        Sp_R = np.kron(np.identity(2), self.S_m)
        Sp_C = Sp_L - Sp_R
        
        Sm_L = np.kron(self.S_m, np.identity(2))
        Sm_R = np.kron(np.identity(2), self.S_p)
        Sm_C = Sm_L - Sm_R
        
        R_S = 1 * ((1 / t2e) - 0.5 * (1 / t1e)) * (Sz_C @ Sz_C) + 1 * 0.25 * (1 / t1e) * ((Sp_C @ Sm_C) + (Sm_C @ Sp_C))
        R_th = 1 * 0.5 * self.p_0 * (1 / t1e) * ((Sp_L @ Sm_R) - (Sm_L @ Sp_R))\
            + 1 * 0.5 * self.p_0 * (1 / t1e) * (Sz_L + Sz_R)
            
        Rtot = R_S + R_th
        return Rtot

    "Time evolve density operator"
    def evolve(self):
        "Define time scale"
        N = 1000 # num of points
        End = 1e-3 # end time
        T = np.linspace(0, End, N)
        delta_t = End / N
        
        "Preallocate matrix for magnetisations"
        M_e = np.zeros([4, T.size])
        
        "Calculate relaxation Liouvillian"
        self.R = self.relax()


        for n in range(N):
            "Calculate x,y,z magnetisations by tracing density operator with Pauli matrices"
            for a in[1,2,3]:
                M_e[a,n] =np.trace(self.S[a] @ self.Density_Operator)


            "Evolve Density Operator in Liouville space"
            self.Density_Operator = np.ravel(self.Density_Operator) # convert to vector
            
            L = self.L_H - 1.j * self.R
            
            self.Density_Operator = linalg.expm(-1.j * L * delta_t) @ self.Density_Operator # LV equation   
            
            self.Density_Operator = self.Density_Operator.reshape(2,2) # convert to matrix
    
        "Calculate total magnetisation"
        M_e[0,:] = ((M_e[1,:]**2)+(M_e[2,:]**2)+(M_e[3,:]**2))**0.5
        

        "Plot x,y,z and total magnetisations for nucleus and electron"
        fige, ((axe1, axe2), (axe3, axe4)) = plt.subplots(2, 2)
        fige.suptitle('Electron Magnetisation', fontsize = 15)
        
        "x magnetisation"
        axe1.plot(T, M_e[1, :], 'k')
        axe1.set(xlabel='t', ylabel='$M_x$')

        "y magnetisation"
        axe2.plot(T, M_e[2, :], 'k')
        axe2.set(xlabel='t', ylabel='$M_y$')

        "z magnetisation"
        axe3.plot(T, M_e[3, :], 'k')
        axe3.set(xlabel='t', ylabel='$M_z$')

        "total magnetisation"
        axe4.plot(T, M_e[0, :], 'k')
        axe4.set(xlabel='t', ylabel='M')
        
        "Plot logarithm of total magnetisation envelope to find decay rate"
        x = T
        y = np.log(M_e[0,:])
        
        self.c = np.polyfit(x, y, 1) # [gradient, intercept] tuple
        R = 1 / self.c[0] # relaxation rate
        print('Relaxation rate is :', R)
        plt.figure()
        plt.plot(x, y, 'k', linewidth = 4)
        plt.plot(x, self.c[0] * x + self.c[1], 'r', linewidth = 1)


        
"electron Larmor frequency"
w_e = 263e9 

"Relaxation rates"
t1e = 0.3e-3
t2e = 0.1e-6

"Generate and evolve system"
x = system(w_e)
x.evolve()