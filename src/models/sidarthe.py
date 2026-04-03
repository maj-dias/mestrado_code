"""
Modelo SIDARTHE para COVID-19

Este módulo implementa o modelo SIDARTHE (Susceptible-Infected-Diagnosed-Ailing-
Recognized-Threatened-Healed-Extinct) desenvolvido especificamente para modelar
a pandemia de COVID-19.

Referência:
Giordano, G., Blanchini, F., Bruno, R., Colaneri, P., Di Filippo, A.,
Di Matteo, A., & Colaneri, M. (2020).
"Modelling the COVID-19 epidemic and implementation of population-wide
interventions in Italy"
Nature Medicine, 26(6), 855-860.
DOI: 10.1038/s41591-020-0883-7

Compartimentos (8 estados):
- S: Susceptible (Suscetíveis)
- I: Infected (Infectados assintomáticos não detectados)
- D: Diagnosed (Infectados assintomáticos detectados)
- A: Ailing (Sintomáticos não detectados)
- R: Recognized (Sintomáticos detectados)
- T: Threatened (Casos graves - hospitalizados/UTI)
- H: Healed (Curados)
- E: Extinct (Mortos)

Equações diferenciais:
    dS/dt = -S·[α(I+D) + β(A+R)]/N
    dI/dt = S·[α(I+D) + β(A+R)]/N - (ε+ζ+λ)I
    dD/dt = εI - (λ+ρ)D
    dA/dt = ζI - (η+μ)A
    dR/dt = ηA - (θ+κ+ρ)R
    dT/dt = θD + κR - (ν+τ)T
    dH/dt = λI + ρD + νT + μA + ρR
    dE/dt = τT

Parâmetros (12 no total):
- α: Taxa de transmissão por assintomáticos (I, D)
- β: Taxa de transmissão por sintomáticos (A, R)
- ε: Taxa de detecção de assintomáticos (I → D)
- ζ: Taxa de desenvolvimento de sintomas (I → A)
- η: Taxa de detecção de sintomáticos (A → R)
- λ: Taxa de cura de infectados não detectados (I → H)
- μ: Taxa de cura de sintomáticos não detectados (A → H)
- ρ: Taxa de cura de detectados (D → H, R → H)
- θ: Taxa de agravamento de assintomáticos detectados (D → T)
- κ: Taxa de agravamento de sintomáticos detectados (R → T)
- ν: Taxa de cura de casos graves (T → H)
- τ: Taxa de mortalidade (T → E)
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Tuple, Dict, Optional


class SIDARTHEModel:
    """
    Modelo SIDARTHE com 8 compartimentos para COVID-19

    Este modelo distingue entre casos detectados e não detectados,
    assintomáticos e sintomáticos, e modela explicitamente hospitalizações
    e mortes.

    Parâmetros
    ----------
    params : Dict[str, float]
        Dicionário com os 12 parâmetros do modelo:
        - alpha: Taxa de transmissão por assintomáticos
        - beta: Taxa de transmissão por sintomáticos
        - epsilon: Taxa de detecção de assintomáticos
        - zeta: Taxa de desenvolvimento de sintomas
        - eta: Taxa de detecção de sintomáticos
        - lambda_: Taxa de cura de não detectados
        - mu: Taxa de cura de sintomáticos não detectados
        - rho: Taxa de cura de detectados
        - theta: Taxa de agravamento D→T
        - kappa: Taxa de agravamento R→T
        - nu: Taxa de cura de casos graves
        - tau: Taxa de mortalidade
    N : float
        População total
    """

    def __init__(self, params: Dict[str, float], N: float):
        # Parâmetros de transmissão
        self.alpha = params['alpha']
        self.beta = params['beta']

        # Parâmetros de progressão
        self.epsilon = params['epsilon']
        self.zeta = params['zeta']
        self.eta = params['eta']

        # Parâmetros de cura
        self.lambda_ = params['lambda_']
        self.mu = params['mu']
        self.rho = params['rho']

        # Parâmetros de agravamento
        self.theta = params['theta']
        self.kappa = params['kappa']

        # Parâmetros de desfecho
        self.nu = params['nu']
        self.tau = params['tau']

        self.N = N
        self.validate_parameters()

    def validate_parameters(self):
        """Valida que todos os parâmetros são positivos e plausíveis"""
        params_dict = {
            'alpha': self.alpha,
            'beta': self.beta,
            'epsilon': self.epsilon,
            'zeta': self.zeta,
            'eta': self.eta,
            'lambda': self.lambda_,
            'mu': self.mu,
            'rho': self.rho,
            'theta': self.theta,
            'kappa': self.kappa,
            'nu': self.nu,
            'tau': self.tau,
            'N': self.N
        }

        for name, value in params_dict.items():
            if value < 0:
                raise ValueError(f"Parâmetro '{name}' deve ser não-negativo, recebido: {value}")

        if self.N <= 0:
            raise ValueError(f"População N deve ser positiva, recebido: {self.N}")

        # Validações de plausibilidade
        if self.alpha > 1.0:
            raise ValueError(f"alpha (transmissão assintomáticos) muito alto: {self.alpha}")
        if self.beta > 1.0:
            raise ValueError(f"beta (transmissão sintomáticos) muito alto: {self.beta}")
        if self.tau > 0.1:
            raise ValueError(f"tau (mortalidade) implausível: {self.tau} (>10% por dia)")

    def derivatives(self, t: float, y: np.ndarray) -> np.ndarray:
        """
        Calcula as derivadas do sistema SIDARTHE

        Parâmetros
        ----------
        t : float
            Tempo atual
        y : np.ndarray
            Vetor de estado [S, I, D, A, R, T, H, E] (8 dimensões)

        Retorna
        -------
        np.ndarray
            Vetor de derivadas [dS/dt, dI/dt, dD/dt, dA/dt, dR/dt, dT/dt, dH/dt, dE/dt]
        """
        S, I, D, A, R, T, H, E = y

        # Taxa de infecção
        # Assintomáticos (I+D) transmitem com taxa α
        # Sintomáticos (A+R) transmitem com taxa β
        infection_rate = (self.alpha * (I + D) + self.beta * (A + R)) / self.N

        # Equações diferenciais
        dS = -S * infection_rate

        dI = (S * infection_rate
              - (self.epsilon + self.zeta + self.lambda_) * I)

        dD = self.epsilon * I - (self.lambda_ + self.rho) * D

        dA = self.zeta * I - (self.eta + self.mu) * A

        dR = self.eta * A - (self.theta + self.kappa + self.rho) * R

        dT = self.theta * D + self.kappa * R - (self.nu + self.tau) * T

        dH = (self.lambda_ * I + self.rho * D + self.nu * T +
              self.mu * A + self.rho * R)

        dE = self.tau * T

        return np.array([dS, dI, dD, dA, dR, dT, dH, dE])

    def simulate(
        self,
        y0: np.ndarray,
        t_span: Tuple[float, float],
        t_eval: np.ndarray,
        method: str = 'RK45'
    ) -> Dict[str, np.ndarray]:
        """
        Simula a evolução temporal do modelo SIDARTHE

        Parâmetros
        ----------
        y0 : np.ndarray
            Condições iniciais [S0, I0, D0, A0, R0, T0, H0, E0]
        t_span : Tuple[float, float]
            Intervalo de tempo (t_inicio, t_fim)
        t_eval : np.ndarray
            Pontos de tempo específicos onde avaliar a solução
        method : str, default='RK45'
            Método de integração numérica

        Retorna
        -------
        dict
            Dicionário com 't', 'S', 'I', 'D', 'A', 'R', 'T', 'H', 'E'
        """
        # Validar condições iniciais
        if len(y0) != 8:
            raise ValueError(f"Condições iniciais devem ter 8 elementos, recebido: {len(y0)}")

        if any(y0 < 0):
            raise ValueError("Condições iniciais não podem ser negativas")

        # Resolver sistema de ODEs
        solution = solve_ivp(
            fun=lambda t, y: self.derivatives(t, y),
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method=method,
            rtol=1e-8,
            atol=1e-10
        )

        if not solution.success:
            raise RuntimeError(f"Integração falhou: {solution.message}")

        return {
            't': solution.t,
            'S': solution.y[0],
            'I': solution.y[1],
            'D': solution.y[2],
            'A': solution.y[3],
            'R': solution.y[4],
            'T': solution.y[5],
            'H': solution.y[6],
            'E': solution.y[7]
        }

    def compute_R0(self) -> float:
        """
        Calcula o número básico de reprodução R₀ para SIDARTHE

        Nota: A fórmula exata requer análise de matriz de próxima geração.
        Esta é uma aproximação simplificada.

        Para análise completa, ver:
        - van den Driessche & Watmough (2002)
        - Giordano et al. (2020) supplementary material

        Aproximação:
        R₀ ≈ (α + β) / taxa_média_remoção

        Retorna
        -------
        float
            Número básico de reprodução (aproximado)
        """
        # Taxa média de remoção do compartimento I
        # (assintomáticos não detectados são principal fonte de infecção inicial)
        removal_rate_I = self.epsilon + self.zeta + self.lambda_

        # Aproximação simplificada
        # Fórmula completa requer eigenvalues da matriz de próxima geração
        R0_approx = (self.alpha + self.beta) / removal_rate_I

        return R0_approx

    def compute_observables(self, solution: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calcula observáveis que podem ser comparados com dados reais

        Parâmetros
        ----------
        solution : dict
            Solução retornada pelo método simulate()

        Retorna
        -------
        dict
            Dicionário com observáveis:
            - 'confirmed': Casos confirmados totais (D+R+T+H+E)
            - 'active': Casos ativos (D+R+T)
            - 'hospitalized': Hospitalizados (T)
            - 'deaths': Mortes acumuladas (E)
            - 'recovered': Recuperados (H)
        """
        D = solution['D']
        R = solution['R']
        T = solution['T']
        H = solution['H']
        E = solution['E']

        return {
            't': solution['t'],
            'confirmed': D + R + T + H + E,
            'active': D + R + T,
            'hospitalized': T,
            'deaths': E,
            'recovered': H
        }

    def __repr__(self) -> str:
        R0 = self.compute_R0()
        return (f"SIDARTHEModel(\n"
                f"  Transmissão: α={self.alpha:.4f}, β={self.beta:.4f}\n"
                f"  Detecção: ε={self.epsilon:.4f}, η={self.eta:.4f}\n"
                f"  Progressão: ζ={self.zeta:.4f}\n"
                f"  Cura: λ={self.lambda_:.4f}, μ={self.mu:.4f}, ρ={self.rho:.4f}\n"
                f"  Agravamento: θ={self.theta:.4f}, κ={self.kappa:.4f}\n"
                f"  Desfecho: ν={self.nu:.4f}, τ={self.tau:.4f}\n"
                f"  População N={self.N:.0f}\n"
                f"  R₀ (aprox) = {R0:.4f}\n"
                f")")


class SIDARTHEControlledModel:
    """
    SIDARTHE model with social distancing (u1) and vaccination (u2) controls.

    States x = [S, I, D, A, Rs, T, H, E]
    (Rs = Recognized compartment, index 4)

    dS/dt  = -S*(1-u1)*(alpha*(I+D) + beta*(A+Rs))/N - u2
    dI/dt  =  S*(1-u1)*(alpha*(I+D) + beta*(A+Rs))/N - (eps+zeta+lam_r)*I
    dD/dt  =  eps*I - (lam_r+rho)*D
    dA/dt  =  zeta*I - (eta+mu)*A
    dRs/dt =  eta*A - (theta+kappa+rho)*Rs
    dT/dt  =  theta*D + kappa*Rs - (nu+tau)*T
    dH/dt  =  lam_r*I + rho*D + nu*T + mu*A + rho*Rs + u2
    dE/dt  =  tau*T

    Controls
    --------
    u1(t): social distancing, reduces alpha and beta by (1-u1)
    u2(t): vaccination flow [persons/day], moves S -> H
    """

    def __init__(self, params: dict, N: float):
        self.alpha = params['alpha']
        self.beta_s = params['beta']
        self.epsilon = params['epsilon']
        self.zeta = params['zeta']
        self.eta = params['eta']
        self.lambda_r = params['lambda_r']
        self.mu = params['mu']
        self.rho = params['rho']
        self.theta = params['theta']
        self.kappa = params['kappa']
        self.nu = params['nu']
        self.tau = params['tau']
        self.N = N

    def derivatives(self, t: float, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        S, I, D, A, Rs, T, H, E = y
        u1, u2 = u

        infection_rate = S * (1.0 - u1) * (
            self.alpha * (I + D) + self.beta_s * (A + Rs)
        ) / self.N

        dS = -infection_rate - u2
        dI = infection_rate - (self.epsilon + self.zeta + self.lambda_r) * I
        dD = self.epsilon * I - (self.lambda_r + self.rho) * D
        dA = self.zeta * I - (self.eta + self.mu) * A
        dRs = self.eta * A - (self.theta + self.kappa + self.rho) * Rs
        dT = self.theta * D + self.kappa * Rs - (self.nu + self.tau) * T
        dH = (self.lambda_r * I + self.rho * D + self.nu * T
              + self.mu * A + self.rho * Rs + u2)
        dE = self.tau * T

        return np.array([dS, dI, dD, dA, dRs, dT, dH, dE])

    def simulate_with_control(
        self,
        initial_conditions: np.ndarray,
        control_trajectory,
        t_span: tuple,
        t_eval: np.ndarray,
        method: str = 'RK45'
    ) -> dict:
        def ode(t, y):
            u = control_trajectory(t)
            return self.derivatives(t, y, u)

        sol = solve_ivp(ode, t_span, initial_conditions,
                        t_eval=t_eval, method=method,
                        rtol=1e-8, atol=1e-10)
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")

        u1_traj = np.array([control_trajectory(t)[0] for t in sol.t])
        u2_traj = np.array([control_trajectory(t)[1] for t in sol.t])
        return {
            't': sol.t, 'S': sol.y[0], 'I': sol.y[1], 'D': sol.y[2],
            'A': sol.y[3], 'Rs': sol.y[4], 'T': sol.y[5],
            'H': sol.y[6], 'E': sol.y[7],
            'u1': u1_traj, 'u2': u2_traj
        }

    def compute_R0(self) -> float:
        removal_rate_I = self.epsilon + self.zeta + self.lambda_r
        return (self.alpha + self.beta_s) / removal_rate_I

    def __repr__(self) -> str:
        return (f"SIDARTHEControlledModel("
                f"alpha={self.alpha:.4f}, beta={self.beta_s:.4f}, "
                f"N={self.N:.0f}, R0={self.compute_R0():.4f})")
