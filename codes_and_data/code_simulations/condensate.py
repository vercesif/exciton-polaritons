import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.constants import hbar, e, m_e
from numpy import linalg as LA
import matplotlib.pyplot as plt
import os

class condensate:
    def __init__(self, N, eta, noise_type, p, N_spot, N_sig, pump_type, mass, dispersion_type, gam_0, gam_2, gam_type, R, gam_R, mu_th, g, psi_in): 

        # Units
        self.l_0 = 4.4 
        self.t_0 = 1e-12
        self.e_0 = (2 * np.pi) * hbar / self.t_0 * (1000/e)
        
        # Space
        self.N = N
        self.x = np.arange(self.N)

        self.k = (2 * np.pi) * fftfreq(self.N, 1)
        
        # Dispersion
        self.mass = -3.3e-6 * m_e * mass
        self.E_2 = (1000/e) * (1e6)**2 * ( hbar**2 / (2 * self.mass)) / (self.e_0 * self.l_0**2)
        self.dispersion_type = dispersion_type
        
        # Losses
        self.gam_0 = gam_0 * 48.5 * 0.001 / self.e_0
        self.gam_2 = gam_2 * 1.6e4 * 0.001 / (self.e_0 * self.l_0**2)
        self.gam_type = gam_type
        
        # Coupling with reservoir
        self.R = R * (8.8e-4) / self.l_0
        self.gam_R = gam_R * 0.45 * self.gam_0
        
        # Two-body interactions
        self.mu_th = mu_th / self.e_0
        self.g_R = 0.5 * self.mu_th * self.R / self.gam_0 
        self.g = g * self.g_R
        
        # Pump
        self.pth = self.gam_0 * self.gam_R / self.R
        self.p = p
        self.pump_type = pump_type
        if self.pump_type == 'spot':
            self.N_spot, self.N_sig = N_spot, N_sig
            L_spot = self.N_spot
            L_sig = self.N_sig
            self.P = self.p * self.pth * (1 + np.tanh((L_spot + self.x - self.N/2) / L_sig)) * (1 + np.tanh((L_spot - self.x + self.N/2) / L_sig)) / (1 + np.tanh((L_spot / L_sig)))**2
        elif self.pump_type == 'homo':
            self.P = p * self.pth * np.ones(self.N)
            
        # Noise
        self.sigma = 0.5 * self.gam_0 * np.ones(self.N) 
        self.eta = eta
        self.noise_type = noise_type

        # Initial condition for condensate and reservoir
        self.rho = (self.p-1) * self.gam_R / self.R
        if psi_in == 'statio':
            self.psi_x = np.ones(self.N, dtype=complex) * np.sqrt(self.rho)
        else:
            self.psi_x = psi_in
        #self.n_R = np.zeros(self.N)
        self.n_R = self.P / (self.gam_R + self.R * np.abs(self.psi_x)**2)
        
    # Kinetic terms	
    def E(self, k):
        J = - self.E_2 
        E0 = - 2 * J
        if self.dispersion_type == 'cos':
            return E0 + 2*J * np.cos(k)
        elif self.dispersion_type == 'exp_fit':	
            return self.E_2 * k**2 
        elif self.dispersion_type == 'par':	
            return self.E_2 * k**2 
        
    def Gam(self, k):
        gam_sat = 2 * (1 + 1/1.18) * self.gam_0
        alpha = self.gam_0 / (gam_sat - self.gam_0)
        beta = self.gam_2 * (1 + alpha) / self.gam_0
        if self.gam_type == 'sat':
                return self.gam_0 * (1 + alpha) / (np.exp(-beta * k**2) + alpha)
        elif self.gam_type == 'par':
            return self.gam_0 + self.gam_2 * k**2
        
    # Gross-Pitaevskii equation terms
    def Linear(self):
        return self.E(self.k) - 1j * 0.5 * self.Gam(self.k)
    
    def NonLinear(self):
        return self.g * np.abs(self.psi_x)**2 + 2 * self.g_R * self.n_R + 1j * 0.5 * self.R * self.n_R
    
    def Reservoir(self):
        return self.P - (self.gam_R + self.R * np.abs(self.psi_x)**2) * self.n_R
    
    def f_Res(self, nR):
        return self.P - (self.gam_R + self.R * np.abs(self.psi_x)**2) * nR
    
    def Reservoir_RK4(self, dt):
        k1 = self.f_Res(self.n_R)
        k2 = self.f_Res(self.n_R + k1 * 0.5 * dt)
        k3 = self.f_Res(self.n_R + k2 * 0.5 * dt)
        k4 = self.f_Res(self.n_R + k3 * dt)
        return (k1 + 2*k2 + 2*k3 + k4) * (dt / 6)
        
    # Steps of time evolution
    def Deterministic_Adiabatic(self, dt):
        self.n_R = self.P / (self.gam_R + self.R * np.abs(self.psi_x)**2)
        self.psi_x = self.psi_x * np.exp(-1j * 0.5 * dt * self.NonLinear())
        psi_k = fft(self.psi_x) * np.exp(-1j * dt * self.Linear())
        self.psi_x = ifft(psi_k) * np.exp(-1j * 0.5 * dt * self.NonLinear())
    
    def Deterministic(self, dt):
        self.n_R = self.n_R + dt * self.Reservoir()
        self.psi_x = self.psi_x * np.exp(-1j * 0.5 * dt * self.NonLinear())
        psi_k = fft(self.psi_x) * np.exp(-1j * dt * self.Linear())
        self.psi_x = ifft(psi_k) * np.exp(-1j * 0.5 * dt * self.NonLinear())
        
    def Deterministic_RK4(self, dt):
        self.n_R = self.n_R + self.Reservoir_RK4(dt)
        self.psi_x = self.psi_x * np.exp(-1j * 0.5 * dt * self.NonLinear())
        psi_k = fft(self.psi_x) * np.exp(-1j * dt * self.Linear())
        self.psi_x = ifft(psi_k) * np.exp(-1j * 0.5 * dt * self.NonLinear())
        
    def Stochastic(self, dt):
        if self.noise_type == 'res':
            self.sigma = 0.5 * self.R * self.n_R
        self.psi_x += np.sqrt(self.eta * dt * self.sigma / 2) * (np.random.normal(0, 1, self.N) + 1j * np.random.normal(0, 1, self.N))
        
    # Evolution
    def Evolution(self, t_fin, dt, N_sample, save_time, reservoir_evolution):
        psi = []
        Nsteps = int(t_fin / dt)
        np.random.seed()
        for i in range(1, Nsteps+1):
            if reservoir_evolution=='Euler': self.Deterministic(dt)
            elif reservoir_evolution=='RK4': self.Deterministic_RK4(dt)
            elif reservoir_evolution=='Adiabatic': self.Deterministic_Adiabatic(dt)
          
            self.Stochastic(dt)
            if i % N_sample == 0:
                if save_time == True: 
                    psi.append(self.psi_x)
        if save_time == True:
            return np.array(psi)
        else:
            return self.psi_x


    def Evolution_nR(self, t_fin, dt, N_sample, save_time, reservoir_evolution):
        psi = []
        nR = []
        Nsteps = int(t_fin / dt)
        np.random.seed()
        for i in range(1, Nsteps+1):
            if reservoir_evolution=='Euler': self.Deterministic(dt)
            elif reservoir_evolution=='RK4': self.Deterministic_RK4(dt)
            elif reservoir_evolution=='Adiabatic': self.Deterministic_Adiabatic(dt)
          
            self.Stochastic(dt)
            if i % N_sample == 0:
                if save_time == True: 
                    psi.append(self.psi_x)
                    nR.append(self.n_R)
        if save_time == True:
            return np.array(psi), np.array(nR)
        else:
            return self.psi_x, self.n_R
    
    def Evolution_noiseless(self, t_fin, dt, N_sample, save_time, reservoir_evolution):
        psi = []
        Nsteps = int(t_fin / dt)
        for i in range(1, Nsteps+1):
            if reservoir_evolution=='Euler': self.Deterministic(dt)
            elif reservoir_evolution=='RK4': self.Deterministic_RK4(dt)
            elif reservoir_evolution=='Adiabatic': self.Deterministic_Adiabatic(dt)
            if i % N_sample == 0:
                if save_time == True: 
                    psi.append(self.psi_x)
        if save_time == True:
            return np.array(psi)
        else:
            return self.psi_x
        
        
    # Bogoliubov (uniform pump)
    def Bogoliubov(self, n_modes):
        rho = (self.p-1) * self.gam_R / self.R
        n_0 = self.gam_0 / self.R
        k = 2 * np.pi * fftfreq(n_modes, self.N/n_modes)
        #k = fftfreq(n_modes, self.N/n_modes)
        nk = len(k[k>=0])
        #a_k = self.E(self.k[self.k>=0]) - 1j * 0.5 * self.Gam(self.k[self.k>=0]) + self.g * rho + 1j * 0.5 * self.gam_0
        a_k = self.E(k[k>=0]) - 1j * 0.5 * self.Gam(k[k>=0]) + self.g * rho + 1j * 0.5 * self.gam_0
        b = self.g * rho
        c = n_0 * (2 * self.g_R + 1j * 0.5 * self.R)
        d = -1j * self.R * rho
        f = -1j * (self.gam_R + self.R * rho)
        l_1, l_2, l_3  = np.zeros(nk, dtype='complex'), np.zeros(nk, dtype='complex'), np.zeros(nk, dtype='complex')

        for i in range(len(a_k)):
            L = np.array([a_k[i],b,c,-np.conjugate(b), -np.conjugate(a_k[i]),-np.conjugate(c),d,d,f])
            L = np.reshape(L, (3,3))
            l_1[i], l_2[i], l_3[i] = LA.eigvals(L)
        return k[k>=0], l_1, l_2, l_3
    
    # KPZ Mapping (uniform pump)
    def KPZ_Mapping(self, rho_fac):
        rho = (self.p-1)*self.gam_R/self.R	
        rho *= rho_fac
        g_eff = self.g - 2 * self.g_R * (self.p*self.pth * self.R) / (self.R * rho + self.gam_R)**2
        a = g_eff * 2 * (self.R * rho + self.gam_R)**2 / ( self.R**2 * self.p * self.pth)
        
        nu = a * self.E_2 + 0.5 * self.gam_2 
        lam = - 2 * (self.E_2 - 0.5 * a * self.gam_2)
        #D = 0.5 * (1 + a*a) * self.eta * 2 *self.sigma[self.N//2] / rho
        D = 0.5 * (1 + a*a) * self.eta * np.sqrt(2) * self.sigma[self.N//2] / rho
        #D = 0.5 * (1 + a*a) * self.eta *self.sigma[self.N//2] / rho

        return nu, lam, D
    
    def nk_bogoliubov(self, k, matrix_type):
        if matrix_type=='Adiabatic':
            #ek = 0.5* self.E_2 * k**2
            ek = self.E(k)
            mu_eff = (self.g - 2 * self.g_R * self.gam_0 / (self.gam_R * self.p)) * self.rho
            Ek = np.sqrt(ek * (ek + 2*mu_eff))
            G0 = 0.5 * self.gam_0 * (self.p-1) / self.p 
            Gk = 0.5 * self.gam_2 * k**2 

            nk = np.zeros_like(k)
            nk[k!=0] = (self.sigma[0] / (G0 + Gk[k!=0]) ) * (1 + (G0**2 + mu_eff**2) / (Ek[k!=0]**2 + Gk[k!=0]**2 + 2*G0*Gk[k!=0]))
            nk[k==0] = 0

        return nk
