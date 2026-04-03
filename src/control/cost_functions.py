"""
Cost Functions for Optimal Control

This module defines cost functionals for epidemic optimal control problems.

Classes
-------
QuadraticCost
    Original quadratic cost for SEIR (arbitrary weights).
RealisticCostSEIR
    Economically grounded cost for SEIR (Brazil 2020 data).
RealisticCostSIR
    Economically grounded cost for SIR (Brazil 2020 data).
RealisticCostSIDARTHE
    Economically grounded cost for SIDARTHE (Brazil 2020 data).

Cost structure (all realistic classes)
---------------------------------------
    L(x, u, t) = c_health * x_health(t)
               + c_G * u1(t)  +  (r1/2) * u1^2(t)
               + c_V * u2(t)  +  (r2/2) * u2^2(t)

where c_health, c_G, c_V are derived from real economic/epidemiological data.
"""

import numpy as np


# =============================================================================
# Original quadratic cost (kept for backward compatibility)
# =============================================================================

class QuadraticCost:
    """
    Quadratic cost function for SEIR optimal control (arbitrary weights).

    L(x, u, t) = w1*I + w2*u1^2 + w3*u2^2
    Phi(x(T))  = wf*I(T)
    """

    def __init__(self, w1: float, w2: float, w3: float, wf: float):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.wf = wf

    def running_cost(self, x: np.ndarray, u: np.ndarray, t: float) -> float:
        S, E, I, R = x
        u1, u2 = u
        return self.w1 * I + self.w2 * u1**2 + self.w3 * u2**2

    def terminal_cost(self, x: np.ndarray) -> float:
        S, E, I, R = x
        return self.wf * I

    def terminal_costates(self, x: np.ndarray) -> np.ndarray:
        return np.array([0.0, 0.0, self.wf, 0.0])

    def __repr__(self) -> str:
        return (f"QuadraticCost(w1={self.w1:.2e}, w2={self.w2:.2e}, "
                f"w3={self.w3:.2e}, wf={self.wf:.2e})")


# =============================================================================
# Realistic cost — SEIR
# =============================================================================

class RealisticCostSEIR:
    """
    Economically grounded cost function for SEIR with Pontryagin conditions.

    Running cost (R$/day):
        L = c_I * I(t)
          + c_G * u1(t) + (r1/2) * u1(t)^2
          + c_V * u2(t) + (r2/2) * u2(t)^2

    Terminal cost (R$):
        Phi(x(T)) = (VSL * IFR) * I(T)

    Coefficients derived from Brazil 2020 data:
        c_I = (VSL * IFR + c_H * h) * gamma   [R$/infected/day]
        c_G = GDP_daily * alpha_GDP            [R$/day/lockdown unit]
        c_V = cost per person vaccinated       [R$/person]

    Model structure assumed:
        dS/dt = -beta*(1-u1)*S*I/N - u2
        dE/dt =  beta*(1-u1)*S*I/N - sigma*E
        dI/dt =  sigma*E - gamma*I
        dR/dt =  gamma*I + u2
    """

    def __init__(self,
                 beta: float, sigma: float, gamma: float, N: float,
                 VSL: float = 5e6, IFR: float = 0.015,
                 h: float = 0.05, c_H: float = 50_000,
                 GDP_daily: float = 20.3e9, alpha_GDP: float = 0.5,
                 c_vaccine: float = 75.0,
                 r1: float = 1e7, r2: float = 0.1):

        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

        self.c_I = (VSL * IFR + c_H * h) * gamma
        self.c_G = GDP_daily * alpha_GDP
        self.c_V = c_vaccine
        self.r1 = r1
        self.r2 = r2
        self.terminal_coeff = VSL * IFR

    def running_cost(self, x: np.ndarray, u: np.ndarray, t: float) -> float:
        S, E, I, R = x
        u1, u2 = u
        return (self.c_I * I
                + self.c_G * u1 + 0.5 * self.r1 * u1**2
                + self.c_V * u2 + 0.5 * self.r2 * u2**2)

    def terminal_cost(self, x: np.ndarray) -> float:
        S, E, I, R = x
        return self.terminal_coeff * I

    def terminal_costates(self, x: np.ndarray) -> np.ndarray:
        """p(T) = dPhi/dx = [0, 0, VSL*IFR, 0]"""
        return np.array([0.0, 0.0, self.terminal_coeff, 0.0])

    def costate_derivatives(self,
                            x: np.ndarray,
                            u: np.ndarray,
                            p: np.ndarray) -> np.ndarray:
        """
        dp/dt = -dH/dx for SEIR with realistic cost.

        dp_S/dt = beta*(1-u1)*I/N * (p_S - p_E)
        dp_E/dt = sigma * (p_E - p_I)
        dp_I/dt = -c_I + beta*(1-u1)*S/N*(p_S - p_E) + gamma*(p_I - p_R)
        dp_R/dt = 0
        """
        S, E, I, R = x
        u1, u2 = u
        p_S, p_E, p_I, p_R = p

        force = self.beta * (1.0 - u1) * I / self.N

        dp_S = force * (p_S - p_E)
        dp_E = self.sigma * (p_E - p_I)
        dp_I = (-self.c_I
                + self.beta * (1.0 - u1) * S / self.N * (p_S - p_E)
                + self.gamma * (p_I - p_R))
        dp_R = 0.0

        return np.array([dp_S, dp_E, dp_I, dp_R])

    def optimal_control(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        u1* = -(c_G + beta*S*I/N*(p_S - p_E)) / r1
        u2* = -(c_V + p_R - p_S) / r2  =>  (p_S - c_V)/r2  (p_R=0)
        """
        S, E, I, R = x
        p_S, p_E, p_I, p_R = p

        u1_raw = -(self.c_G + self.beta * S * I / self.N * (p_S - p_E)) / self.r1
        u2_raw = -(self.c_V + p_R - p_S) / self.r2

        return np.array([u1_raw, u2_raw])

    def __repr__(self) -> str:
        return (f"RealisticCostSEIR(c_I={self.c_I:.2e}, c_G={self.c_G:.2e}, "
                f"c_V={self.c_V:.1f}, terminal={self.terminal_coeff:.2e})")


# =============================================================================
# Realistic cost — SIR
# =============================================================================

class RealisticCostSIR:
    """
    Economically grounded cost function for SIR with Pontryagin conditions.

    Running cost (R$/day):
        L = c_I * I(t)
          + c_G * u1(t) + (r1/2) * u1(t)^2
          + c_V * u2(t) + (r2/2) * u2(t)^2

    Terminal cost (R$):
        Phi(x(T)) = (VSL * IFR) * I(T)

    c_I = (VSL * IFR + c_H * h) * gamma_SIR

    Note: gamma_SIR absorbs both incubation and infectious periods.
    The total cost per infected individual equals VSL*IFR + c_H*h regardless
    of the specific gamma value.

    Model structure assumed:
        dS/dt = -beta*(1-u1)*S*I/N - u2
        dI/dt =  beta*(1-u1)*S*I/N - gamma*I
        dR/dt =  gamma*I + u2
    """

    def __init__(self,
                 beta: float, gamma: float, N: float,
                 VSL: float = 5e6, IFR: float = 0.015,
                 h: float = 0.05, c_H: float = 50_000,
                 GDP_daily: float = 20.3e9, alpha_GDP: float = 0.5,
                 c_vaccine: float = 75.0,
                 r1: float = 1e7, r2: float = 0.1):

        self.beta = beta
        self.gamma = gamma
        self.N = N

        self.c_I = (VSL * IFR + c_H * h) * gamma
        self.c_G = GDP_daily * alpha_GDP
        self.c_V = c_vaccine
        self.r1 = r1
        self.r2 = r2
        self.terminal_coeff = VSL * IFR

    def running_cost(self, x: np.ndarray, u: np.ndarray, t: float) -> float:
        S, I, R = x
        u1, u2 = u
        return (self.c_I * I
                + self.c_G * u1 + 0.5 * self.r1 * u1**2
                + self.c_V * u2 + 0.5 * self.r2 * u2**2)

    def terminal_cost(self, x: np.ndarray) -> float:
        S, I, R = x
        return self.terminal_coeff * I

    def terminal_costates(self, x: np.ndarray) -> np.ndarray:
        """p(T) = dPhi/dx = [0, VSL*IFR, 0]"""
        return np.array([0.0, self.terminal_coeff, 0.0])

    def costate_derivatives(self,
                            x: np.ndarray,
                            u: np.ndarray,
                            p: np.ndarray) -> np.ndarray:
        """
        dp/dt = -dH/dx for SIR with realistic cost.

        dp_S/dt = beta*(1-u1)*I/N * (p_S - p_I)
        dp_I/dt = -c_I + beta*(1-u1)*S/N*(p_S - p_I) + gamma*(p_I - p_R)
        dp_R/dt = 0
        """
        S, I, R = x
        u1, u2 = u
        p_S, p_I, p_R = p

        force = self.beta * (1.0 - u1) * I / self.N

        dp_S = force * (p_S - p_I)
        dp_I = (-self.c_I
                + self.beta * (1.0 - u1) * S / self.N * (p_S - p_I)
                + self.gamma * (p_I - p_R))
        dp_R = 0.0

        return np.array([dp_S, dp_I, dp_R])

    def optimal_control(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        u1* = -(c_G + beta*S*I/N*(p_S - p_I)) / r1
        u2* = -(c_V + p_R - p_S) / r2  =>  (p_S - c_V)/r2  (p_R=0)
        """
        S, I, R = x
        p_S, p_I, p_R = p

        u1_raw = -(self.c_G + self.beta * S * I / self.N * (p_S - p_I)) / self.r1
        u2_raw = -(self.c_V + p_R - p_S) / self.r2

        return np.array([u1_raw, u2_raw])

    def __repr__(self) -> str:
        return (f"RealisticCostSIR(c_I={self.c_I:.2e}, c_G={self.c_G:.2e}, "
                f"c_V={self.c_V:.1f}, terminal={self.terminal_coeff:.2e})")


# =============================================================================
# Realistic cost — SIDARTHE
# =============================================================================

class RealisticCostSIDARTHE:
    """
    Economically grounded cost function for SIDARTHE with Pontryagin conditions.

    Running cost (R$/day):
        L = c_T_total * T(t)
          + c_G * u1(t) + (r1/2) * u1(t)^2
          + c_V * u2(t) + (r2/2) * u2(t)^2

    where:
        c_T_total = VSL * tau + c_T_icu   [R$/ICU-patient/day]

    Terminal cost (R$):
        Phi(x(T)) = p_T_terminal * T(T)
        p_T_terminal = VSL * tau / (nu + tau)

    Model states x = [S, I, D, A, Rs, T, H, E]:
        dS/dt  = -S*(1-u1)*(alpha*(I+D) + beta*(A+Rs))/N - u2
        dI/dt  =  S*(1-u1)*(alpha*(I+D) + beta*(A+Rs))/N - (eps+zeta+lam_r)*I
        dD/dt  =  eps*I - (lam_r+rho)*D
        dA/dt  =  zeta*I - (eta+mu)*A
        dRs/dt =  eta*A - (theta+kappa+rho)*Rs
        dT/dt  =  theta*D + kappa*Rs - (nu+tau)*T
        dH/dt  =  lam_r*I + rho*D + nu*T + mu*A + rho*Rs + u2
        dE/dt  =  tau*T

    Note: p_H(t) = 0 and p_E(t) = 0 for all t (absorbing states, no future cost).
    """

    def __init__(self,
                 alpha: float, beta: float,
                 epsilon: float, zeta: float, eta: float,
                 lambda_r: float, mu: float, rho: float,
                 theta: float, kappa: float, nu: float, tau: float,
                 N: float,
                 VSL: float = 5e6,
                 c_T_icu: float = 3000.0,
                 GDP_daily: float = 20.3e9, alpha_GDP: float = 0.5,
                 c_vaccine: float = 75.0,
                 r1: float = 1e7, r2: float = 0.1):

        self.alpha = alpha
        self.beta_s = beta      # SIDARTHE beta (symptomatic transmission)
        self.epsilon = epsilon
        self.zeta = zeta
        self.eta = eta
        self.lambda_r = lambda_r
        self.mu = mu
        self.rho = rho
        self.theta = theta
        self.kappa = kappa
        self.nu = nu
        self.tau = tau
        self.N = N

        self.c_T_total = VSL * tau + c_T_icu
        self.c_G = GDP_daily * alpha_GDP
        self.c_V = c_vaccine
        self.r1 = r1
        self.r2 = r2
        self.p_T_terminal = VSL * tau / (nu + tau)

    def running_cost(self, x: np.ndarray, u: np.ndarray, t: float) -> float:
        S, I, D, A, Rs, T, H, E = x
        u1, u2 = u
        return (self.c_T_total * T
                + self.c_G * u1 + 0.5 * self.r1 * u1**2
                + self.c_V * u2 + 0.5 * self.r2 * u2**2)

    def terminal_cost(self, x: np.ndarray) -> float:
        S, I, D, A, Rs, T, H, E = x
        return self.p_T_terminal * T

    def terminal_costates(self, x: np.ndarray) -> np.ndarray:
        """p(T) = [0,0,0,0,0, p_T_terminal, 0, 0]"""
        return np.array([0., 0., 0., 0., 0., self.p_T_terminal, 0., 0.])

    def costate_derivatives(self,
                            x: np.ndarray,
                            u: np.ndarray,
                            p: np.ndarray) -> np.ndarray:
        """
        dp/dt = -dH/dx for SIDARTHE with realistic cost.

        p_H(t) = 0, p_E(t) = 0 for all t.
        """
        S, I, D, A, Rs, T, H, E = x
        u1, u2 = u
        p_S, p_I, p_D, p_A, p_Rs, p_T, p_H, p_E = p

        Gamma = self.alpha * (I + D) + self.beta_s * (A + Rs)
        dF_dS = (1.0 - u1) * Gamma / self.N
        dF_dI = S * (1.0 - u1) * self.alpha / self.N
        dF_dD = S * (1.0 - u1) * self.alpha / self.N
        dF_dA = S * (1.0 - u1) * self.beta_s / self.N
        dF_dRs = S * (1.0 - u1) * self.beta_s / self.N

        dp_S = dF_dS * (p_S - p_I)

        dp_I = (dF_dI * (p_S - p_I)
                + (self.epsilon + self.zeta + self.lambda_r) * p_I
                - self.epsilon * p_D
                - self.zeta * p_A
                - self.lambda_r * p_H)

        dp_D = (dF_dD * (p_S - p_I)
                + (self.lambda_r + self.rho) * p_D
                - self.theta * p_T
                - self.rho * p_H)

        dp_A = (dF_dA * (p_S - p_I)
                + (self.eta + self.mu) * p_A
                - self.eta * p_Rs
                - self.mu * p_H)

        dp_Rs = (dF_dRs * (p_S - p_I)
                 + (self.theta + self.kappa + self.rho) * p_Rs
                 - self.kappa * p_T
                 - self.rho * p_H)

        dp_T = (-self.c_T_total
                + (self.nu + self.tau) * p_T
                - self.nu * p_H
                - self.tau * p_E)

        dp_H = 0.0
        dp_E = 0.0

        return np.array([dp_S, dp_I, dp_D, dp_A, dp_Rs, dp_T, dp_H, dp_E])

    def optimal_control(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        u1* = -(c_G + S*Gamma/N*(p_S - p_I)) / r1
        u2* = -(c_V + p_H - p_S) / r2  =>  (p_S - c_V)/r2  (p_H=0)
        """
        S, I, D, A, Rs, T, H, E = x
        p_S, p_I, p_D, p_A, p_Rs, p_T, p_H, p_E = p

        Gamma = self.alpha * (I + D) + self.beta_s * (A + Rs)

        u1_raw = -(self.c_G + S * Gamma / self.N * (p_S - p_I)) / self.r1
        u2_raw = -(self.c_V + p_H - p_S) / self.r2

        return np.array([u1_raw, u2_raw])

    def __repr__(self) -> str:
        return (f"RealisticCostSIDARTHE("
                f"c_T_total={self.c_T_total:.2e}, c_G={self.c_G:.2e}, "
                f"c_V={self.c_V:.1f}, p_T_terminal={self.p_T_terminal:.2e})")
