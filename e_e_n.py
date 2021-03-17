import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

class system:
    def __init__(self, w):
        "Define Pauli matrices"
        self.sigma_0 = np.array([[1, 0], [0, 1]])
        self.sigma_1 = np.array([[0, 1], [1, 0]])
        self.sigma_2 = np.array([[0, -1.j], [1.j, 0]])
        self.sigma_3 = np.array([[1, 0], [0, -1]])
        self.pauli = [self.sigma_0, self.sigma_1, self.sigma_2, self.sigma_3]
        
        "raising and lowering operators"
        self.p_p = self.pauli[1] + 1.j*self.pauli[2]
        self.p_m = self.pauli[1] - 1.j*self.pauli[2]
        
        "Define microwave and electron offset frequencies"
        w_mw = w[0] - w_n
        w_off1 = w[0] - w_mw
        w_off2 = w[1] - w_mw
        w_1 = 1e6
        A,C = 0,1e6
        D = 1e6

        "Define Hamiltonian in electron rotating frame"
        self.H_z =  w_off1 * np.kron(self.pauli[3],np.kron(self.pauli[0],self.pauli[0])) + w_off2 * np.kron(self.pauli[0],np.kron(self.pauli[3],self.pauli[0])) - w_n * np.kron(self.pauli[0],np.kron(self.pauli[0],self.pauli[3])) # background z field

        self.H_hfa = A * np.kron(self.pauli[3],np.kron(self.pauli[0],self.pauli[3])) + C * (np.kron(self.pauli[3],np.kron(self.pauli[0],self.p_p)) + np.kron(self.pauli[3],np.kron(self.pauli[0],self.p_m))) # electron 1 hyperfine coupling

        self.H_hfb = A * np.kron(self.pauli[0],np.kron(self.pauli[3],self.pauli[3])) + C * (np.kron(self.pauli[0],np.kron(self.pauli[3],self.p_p)) + np.kron(self.pauli[0],np.kron(self.pauli[3],self.p_m))) # electron 2 hyperfine coupling
        
        self.H_d = D * (2* np.kron(self.pauli[3],np.kron(self.pauli[3],self.pauli[0])) - (np.kron(self.p_p,np.kron(self.p_m,self.pauli[0])) + np.kron(self.p_m,np.kron(self.p_p,self.pauli[0])))) # electron dipole coupling

        self.Hamiltonian = self.H_z + self.H_hfa + self.H_hfb + self.H_d # H0
        
        "Define microwave Hamiltonian in electron rotating frame"
        self.H_MW = w_1 * (np.kron(self.pauli[1],np.kron(self.pauli[0],self.pauli[0])) + np.kron(self.pauli[0],np.kron(self.pauli[1],self.pauli[0]))) #microwave field 

        "Diagonalise Hamiltonian"
        self.eigval,self.eigvec = linalg.eig(self.Hamiltonian)
        self.eigvec_i = linalg.inv(self.eigvec)

        "Convert full Hamiltonian to new basis"
        self.Hamiltonian = self.diag(self.Hamiltonian) + self.diag(self.H_MW)

        "Define density operator and convert to Hamiltonian basis"
        self.Density_Operator = (0.5**len(w)) * (np.kron(self.pauli[3],np.kron(self.pauli[0],self.pauli[0])) + np.kron(self.pauli[0],np.kron(self.pauli[3],self.pauli[0])) + np.kron(self.pauli[0],np.kron(self.pauli[0],self.pauli[3])))
        self.Denisty_Operator = self.diag(self.Density_Operator)

        "Convert principle x,y,z Pauli matrices to Hamiltonian basis"
        self.S1,self.S2,self.I = [0,0,0,0],[0,0,0,0],[0,0,0,0]
        for a in[0,1,2,3]:
            self.S1[a] = self.diag(np.kron(self.pauli[a],np.kron(self.pauli[0],self.pauli[0])))
            self.S2[a] = self.diag(np.kron(self.pauli[0],np.kron(self.pauli[a],self.pauli[0])))
            self.I[a] = self.diag(np.kron(self.pauli[0],np.kron(self.pauli[0],self.pauli[a])))

            
            
    "Convert an operator to basis of diagonalised Hamiltonian"
    def diag(self, operator):
        return (self.eigvec_i @ (operator @ self.eigvec))
    
    "Return an operator to Zeeman basis"
    def undiag(self, operator):
        return (self.eigvec @ (operator @ self.eigvec_i))


    "Time evolve density operator"
    def evolve(self):
        "Define time scale"
        N = 1000
        End = 1e-7
        T = np.linspace(0, End, N)
        delta_t = End / N
        
        M_e1,M_e2,M_n = np.zeros([4, T.size]), np.zeros([4, T.size]), np.zeros([4,T.size])

        for n in range(N):
            "Calculate x,y,z magnetisations by tracing density operator with Pauli matrices"
            for a in[1,2,3]:
                M_e1[a,n] =np.trace(self.S1[a] @ self.Density_Operator)
                M_e2[a,n] =np.trace(self.S2[a] @ self.Density_Operator)
                M_n[a,n] =np.trace(self.I[a] @ self.Density_Operator)
                
            "Time evolve density operator"
            self.Density_Operator = linalg.expm(1.j *self.Hamiltonian*delta_t) @ self.Density_Operator @ linalg.expm(-1.j *self.Hamiltonian*delta_t)

            
           
        "Calculate total magnetisation"
        M_e1[0,:] = ((M_e1[1,:]**2)+(M_e1[2,:]**2)+(M_e1[3,:]**2))**0.5
        M_e1 = -M_e1
        M_e2[0,:] = ((M_e2[1,:]**2)+(M_e2[2,:]**2)+(M_e2[3,:]**2))**0.5
        M_e2 = -M_e2
        M_n[0,:] = ((M_n[1,:]**2)+(M_n[2,:]**2)+(M_n[3,:]**2))**0.5

        "Plot x,y,z and total magnetisations for nucleus and electron"
        fige, ((axe1, axe2), (axe3, axe4)) = plt.subplots(2, 2)
        fige.suptitle('Electron 1 Magnetisation', fontsize = 15)
        figE, ((axE1, axE2), (axE3, axE4)) = plt.subplots(2, 2)
        figE.suptitle('Electron 2 Magnetisation', fontsize = 15)
        fign, ((axn1, axn2), (axn3, axn4)) = plt.subplots(2, 2)
        fign.suptitle('Nuclear Magnetisation', fontsize = 15)
        
        "x magnetisation"
        axe1.plot(T, M_e1[1, :], 'k')
        axe1.set(xlabel='t', ylabel='$M_x$')
        
        axE1.plot(T, M_e2[1, :], 'b')
        axE1.set(xlabel='t', ylabel='$M_x$')

        axn1.plot( T, M_n[1, :], 'r')
        axn1.set(xlabel='t', ylabel='$M_x$')

        "y magnetisation"
        axe2.plot(T, M_e1[2, :], 'k')
        axe2.set(xlabel='t', ylabel='$M_y$')

        axE2.plot(T, M_e2[2, :], 'b')
        axE2.set(xlabel='t', ylabel='$M_x$')

        axn2.plot(T, M_n[2, :], 'r')
        axn2.set(xlabel='t', ylabel='$M_y$')

        "z magnetisation"
        axe3.plot(T, M_e1[3, :], 'k')
        axe3.set(xlabel='t', ylabel='$M_z$')

        axE3.plot(T, M_e2[3, :], 'b')
        axE3.set(xlabel='t', ylabel='$M_x$')

        axn3.plot(T, M_n[3, :], 'r')
        axn3.set(xlabel='t', ylabel='$M_z$')
        
        "total magnetisation"
        axe4.plot(T, M_e1[0, :], 'k')
        axe4.set(xlabel='t', ylabel='M')
        
        axE4.plot(T, M_e2[0, :], 'b')
        axE4.set(xlabel='t', ylabel='$M_x$')
        
        axn4.plot(T, M_n[0, :], 'r')
        axn4.set(xlabel='t', ylabel='M')

        #fige.savefig('E')
        #fign.savefig('N')

"electron and nuclear Larmor frequencies"
w_e1, w_n = 395e9, 600e6
w_e2 = w_e1 + w_n
w = np.array([w_e1,w_e2,w_n])

"run system"
x = system(w)
x.evolve()