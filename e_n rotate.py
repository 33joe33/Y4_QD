import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt


class system:
    
    def __init__(self):
        "Define Pauli matrices"
        self.sigma_0 = 0.5 * np.array([[1, 0], [0, 1]])
        self.sigma_1 = 0.5 * np.array([[0, 1], [1, 0]])
        self.sigma_2 = 0.5 * np.array([[0, -1.j], [1.j, 0]])
        self.sigma_3 = 0.5 * np.array([[1, 0], [0, -1]])
        self.pauli = [self.sigma_0, self.sigma_1, self.sigma_2, self.sigma_3]
        
        "raising and lowering operators"
        self.p_p = self.pauli[1] + 1.j*self.pauli[2]
        self.p_m = self.pauli[1] - 1.j*self.pauli[2]
        
        "Define g-anisotropy in lab frame"
        self.g = np.diag([2.0061,2.0021,2.0094])
        self.g = self.g / self.g[2,2] # define g-anisotropy relative to z
        #Omega = np.array([253.6, 105.1, 123.8]) * (np.pi/180) # aritrary euler angles principle-lab frame
        #self.rotate(self.g, Omega)
        self.rotate(self.g, [np.arctan(np.sqrt(2)), 0, 0]) # rotate to magic angle???
        
        "Create principle x,y,z product operators"
        self.S,self.I = [0,0,0,0],[0,0,0,0]
        for a in[0,1,2,3]:
            self.S[a] = 2 * np.kron(self.pauli[a],self.pauli[0])
            self.I[a] = 2 * np.kron(self.pauli[0],self.pauli[a])
        
        "Create initial Density Operator"
        hbar = 1.054e-34
        k_b = 1.38e-23
        T = 100
        beta = (hbar * w_e) / (2*np.pi * k_b * T) 
        p_0 = np.tanh(beta/2)
        self.Density_Operator = (0.5**2) * (2 * np.kron(self.pauli[0], self.pauli[0]) - 2 * p_0 * 2 * np.kron(self.pauli[3],self.pauli[0]))
        
        "Calculate initial electron polarisation"
        self.E_init = np.abs(np.trace(self.S[3] @ self.Density_Operator))
        

    def create_Hamiltonian(self):
        "Calculate electron offset frequency"
        w_off = w_e * self.g[2,2] - w_mw
        
        "Define Hamiltonian in electron rotating frame"
        self.H_z =  w_off * 2 * np.kron(self.pauli[3],self.pauli[0]) - w_n * 2 * np.kron(self.pauli[0],self.pauli[3]) # background z field

        self.H_hf = self.g[2,0] * C *( 2 * np.kron(self.pauli[3],self.p_p) + 2 * np.kron(self.pauli[3],self.p_m)) # hyperfine coupling
        
        self.Hamiltonian = self.H_z + self.H_hf # H0
        
        "Define microwave Hamiltonian"
        self.H_MW = w_1 * 2 * np.kron(self.pauli[1],self.pauli[0]) # microwave field 

        "Diagonalise Hamiltonian"
        self.eigval,self.eigvec = linalg.eig(self.Hamiltonian)
        self.eigvec_i = linalg.inv(self.eigvec)
        
        self.Hamiltonian = self.diag(self.Hamiltonian) + self.diag(self.H_MW)
        
        "Change basis of Density operator to match Hamiltonian"
        self.Density_Operator = self.diag(self.Density_Operator)


     
    "Convert an operator to basis of diagonalised Hamiltonian"
    def diag(self, operator):
        return (self.eigvec_i @ (operator @ self.eigvec))
    
    "Return an operator to Zeeman basis"
    def undiag(self, operator):
        return (self.eigvec @ (operator @ self.eigvec_i))

    "Rotate g-tensor through Euler angles"
    def rotate(self, operator, angle):
        self.g = Euler_Rot_Mat(angle[0], angle[1], angle[2]) @ operator @ Euler_Rot_Mat(-1*angle[0], -1*angle[1], -1*angle[2])


    "Time evolve density operator"
    def evolve(self):
        "Define time scale"
        N = 1000
        End = 0.00025 #1e-7
        T = np.linspace(0, End, N)
        delta_t = End / N
        w_g = 4e3 * 2*np.pi
        
        "Preallocate matrices for mangetisations and eigenvalues"
        M_e, M_n = np.zeros([4, T.size]), np.zeros([4, T.size])
        eig = np.zeros([4, T.size])

        "Time evolve Density Operator and calculate magnetisation changes"
        for n in range(N):
            
            "Calculate x,y,z magnetisations by tracing density operator with Pauli matrices in Zeeman basis"
            for a in[1,2,3]:
                M_e[a,n] =np.trace(self.S[a] @ self.Density_Operator)
                M_n[a,n] =np.trace(self.I[a] @ self.Density_Operator)
             
            "Rotate sample and create new Hamiltonian"
            self.rotate(self.g, [0.0, w_g * delta_t, 0.0]) # euler angles for MAS
            self.create_Hamiltonian() 
  
            "Extract eigenvalues"
            eig[:,n] = self.eigval
                
            "Change basis of Density Operator to match Hamiltonian and evolve in time"
            self.Density_Operator = self.diag(self.Density_Operator)
            self.Density_Operator = linalg.expm(1.j *self.Hamiltonian*delta_t) @ self.Density_Operator @ linalg.expm(-1.j *self.Hamiltonian*delta_t)      
                
            "Return Density Operator to Zeeman basis"
            self.Density_Operator = self.undiag(self.Density_Operator)
            
                 
        "Calculate total magnetisation"
        M_e[0,:] = ((M_e[1,:]**2)+(M_e[2,:]**2)+(M_e[3,:]**2))**0.5
        M_n[0,:] = ((M_n[1,:]**2)+(M_n[2,:]**2)+(M_n[3,:]**2))**0.5
        
        "Scale total relative to initial magnetisation"
        M_e = M_e / self.E_init
        M_n = M_n / self.E_init
        

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

        "Plot eigenvalues of Hamiltonian"
        figeig, ax =plt.subplots(1,1)
        figeig.suptitle("Eigenvalues", fontsize=15)
        ax.plot(T,eig[0,:])
        ax.plot(T, eig[1, :])
        ax.plot(T, eig[2, :])
        ax.plot(T, eig[3, :])
        ax.set(xlabel='t', ylabel='Eigenvalues')
        

"General 3D rotation through euler angles"
def Euler_Rot_Mat(alpha, beta, gamma):
    Rot = np.array([[np.cos(alpha) * np.cos(beta) * np.cos(gamma) - np.sin(alpha) * np.sin(gamma), -np.sin(alpha) * np.cos(gamma) - np.cos(alpha) * np.cos(beta) * np.sin(gamma), np.cos(alpha) * np.sin(beta)],
                  [np.sin(alpha) * np.cos(beta) * np.cos(gamma) + np.cos(alpha) * np.sin(gamma), np.cos(alpha) * np.cos(gamma) - np.sin(alpha) * np.cos(beta) * np.sin(gamma), np.sin(alpha) * np.sin(beta)],
                  [-np.sin(beta) * np.cos(gamma), np.sin(beta) * np.sin(gamma), np.cos(beta)]])
    return Rot
 

   
    
"electron and nuclear angular Larmor frequencies"
w_e, w_n = 263e9 * 2*np.pi, 400e6 * 2*np.pi

"Define microwave frequency"
a = -1
w_mw = w_e + a * w_n

"Define interaction strengths"
w_1 = 0.85e6 * 2*np.pi
C = 3e6 * 2*np.pi

"run system"
x = system()
x.evolve()
