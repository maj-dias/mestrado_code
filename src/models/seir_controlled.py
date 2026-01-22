"""
Modelo SEIR Controlado (Susceptible-Exposed-Infected-Recovered with Control)

Este módulo implementa o modelo SEIR com variáveis de controle:
- u1(t): Intensidade de distanciamento social/lockdown (0 a u1_max)
- u2(t): Taxa de vacinação (pessoas/dia, 0 a u2_max)

Equações diferenciais controladas:
    dS/dt = -β·(1-u1)·S·I/N - u2·S
    dE/dt = β·(1-u1)·S·I/N - σ·E
    dI/dt = σ·E - γ·I
    dR/dt = γ·I + u2·S

Onde:
- u1 reduz a taxa de transmissão efetiva: β_eff = β·(1-u1)
- u2 move suscetíveis diretamente para removidos (vacinação)
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Tuple, Dict, Optional


class SEIRControlledModel:
    """
    Modelo SEIR com controles de lockdown e vacinação

    Parâmetros
    ----------
    beta : float
        Taxa de transmissão (β > 0)
    sigma : float
        Taxa de incubação (σ > 0)
    gamma : float
        Taxa de recuperação (γ > 0)
    N : float
        População total (N > 0)
    """

    def __init__(self, beta: float, sigma: float, gamma: float, N: float):
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = N
        self.validate_parameters()

    def validate_parameters(self):
        """Valida que todos os parâmetros são positivos"""
        if self.beta <= 0:
            raise ValueError(f"beta deve ser positivo, recebido: {self.beta}")
        if self.sigma <= 0:
            raise ValueError(f"sigma deve ser positivo, recebido: {self.sigma}")
        if self.gamma <= 0:
            raise ValueError(f"gamma deve ser positivo, recebido: {self.gamma}")
        if self.N <= 0:
            raise ValueError(f"N (população) deve ser positivo, recebido: {self.N}")

    def derivatives(self, t: float, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calcula as derivadas do sistema SEIR controlado

        Equações:
            dS/dt = -β·(1-u1)·S·I/N - u2
            dE/dt = β·(1-u1)·S·I/N - σ·E
            dI/dt = σ·E - γ·I
            dR/dt = γ·I + u2

        Parâmetros
        ----------
        t : float
            Tempo atual
        y : np.ndarray
            Vetor de estado [S, E, I, R]
        u : np.ndarray
            Vetor de controle [u1, u2]

        Retorna
        -------
        np.ndarray
            Vetor de derivadas [dS/dt, dE/dt, dI/dt, dR/dt]
        """
        S, E, I, R = y
        u1, u2 = u

        # Taxa de infecção reduzida por lockdown
        infection_rate = self.beta * (1 - u1) * S * I / self.N

        # Sistema de equações diferenciais controladas
        dS = -infection_rate - u2
        dE = infection_rate - self.sigma * E
        dI = self.sigma * E - self.gamma * I
        dR = self.gamma * I + u2

        return np.array([dS, dE, dI, dR])

    def simulate_with_control(
        self,
        initial_conditions: np.ndarray,
        control_trajectory: Callable[[float], np.ndarray],
        t_span: Tuple[float, float],
        t_eval: np.ndarray,
        method: str = 'RK45'
    ) -> Dict[str, np.ndarray]:
        """
        Simula a evolução temporal do modelo SEIR com trajetória de controle

        Parâmetros
        ----------
        initial_conditions : np.ndarray
            Condições iniciais [S0, E0, I0, R0]
        control_trajectory : Callable[[float], np.ndarray]
            Função que retorna u(t) = [u1(t), u2(t)] para cada tempo t
        t_span : Tuple[float, float]
            Intervalo de tempo (t_inicio, t_fim)
        t_eval : np.ndarray
            Pontos de tempo específicos onde avaliar a solução
        method : str, default='RK45'
            Método de integração numérica

        Retorna
        -------
        dict
            Dicionário com 't', 'S', 'E', 'I', 'R', 'u1', 'u2'
        """
        # Função auxiliar para solve_ivp (não recebe u como parâmetro)
        def derivatives_with_control(t, y):
            u = control_trajectory(t)
            return self.derivatives(t, y, u)

        # Resolver sistema de ODEs
        solution = solve_ivp(
            fun=derivatives_with_control,
            t_span=t_span,
            y0=initial_conditions,
            t_eval=t_eval,
            method=method,
            dense_output=True
        )

        if not solution.success:
            raise RuntimeError(f"Integração falhou: {solution.message}")

        # Calcular trajetórias de controle
        u1_trajectory = np.array([control_trajectory(t)[0] for t in solution.t])
        u2_trajectory = np.array([control_trajectory(t)[1] for t in solution.t])

        return {
            't': solution.t,
            'S': solution.y[0],
            'E': solution.y[1],
            'I': solution.y[2],
            'R': solution.y[3],
            'u1': u1_trajectory,
            'u2': u2_trajectory
        }

    def compute_R0(self) -> float:
        """
        Calcula o número básico de reprodução R₀ = β / γ

        Note que R₀ não depende de σ (taxa de incubação).
        Com controle, o R₀ efetivo é: R₀_eff = β·(1-u1) / γ

        Retorna
        -------
        float
            Número básico de reprodução sem controle
        """
        return self.beta / self.gamma

    def __repr__(self) -> str:
        R0 = self.compute_R0()
        incubation_period = 1 / self.sigma
        return (f"SEIRControlledModel(beta={self.beta:.4f}, sigma={self.sigma:.4f}, "
                f"gamma={self.gamma:.4f}, N={self.N:.0f}, "
                f"R0={R0:.4f}, incubation_period={incubation_period:.1f} dias)")
