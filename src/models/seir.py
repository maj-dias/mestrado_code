"""
Modelo SEIR (Susceptible-Exposed-Infected-Recovered)

Este módulo implementa o modelo epidemiológico compartimental SEIR.

O modelo SEIR estende o modelo SIR adicionando um compartimento de expostos:
- S (Susceptibles): Indivíduos suscetíveis à infecção
- E (Exposed): Indivíduos expostos mas ainda não infectantes (período de incubação)
- I (Infected): Indivíduos atualmente infectados e infectantes
- R (Recovered): Indivíduos removidos (recuperados ou falecidos)

Equações diferenciais:
    dS/dt = -β * S * I / N
    dE/dt = β * S * I / N - σ * E
    dI/dt = σ * E - γ * I
    dR/dt = γ * I

Onde:
- β (beta): Taxa de transmissão (contatos efetivos por dia)
- σ (sigma): Taxa de incubação (1 / período de incubação em dias)
- γ (gamma): Taxa de recuperação (1 / duração média da infecção)
- N: População total (constante, S + E + I + R = N)
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Tuple, Dict, Optional


class SEIRModel:
    """
    Modelo SEIR para dinâmica epidemiológica

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

    def derivatives(self, t: float, y: List[float]) -> np.ndarray:
        """
        Calcula as derivadas do sistema SEIR

        Equações:
            dS/dt = -β * S * I / N
            dE/dt = β * S * I / N - σ * E
            dI/dt = σ * E - γ * I
            dR/dt = γ * I

        Parâmetros
        ----------
        t : float
            Tempo atual (não usado diretamente, mas necessário para solve_ivp)
        y : List[float]
            Vetor de estado [S, E, I, R]

        Retorna
        -------
        np.ndarray
            Vetor de derivadas [dS/dt, dE/dt, dI/dt, dR/dt]
        """
        S, E, I, R = y

        # Taxa de infecção
        infection_rate = self.beta * S * I / self.N

        # Sistema de equações diferenciais
        dS = -infection_rate
        dE = infection_rate - self.sigma * E
        dI = self.sigma * E - self.gamma * I
        dR = self.gamma * I

        return np.array([dS, dE, dI, dR])

    def simulate(
        self,
        initial_conditions: List[float],
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        method: str = 'RK45'
    ) -> Dict[str, np.ndarray]:
        """
        Simula a evolução temporal do modelo SEIR

        Parâmetros
        ----------
        initial_conditions : List[float]
            Condições iniciais [S0, E0, I0, R0]
        t_span : Tuple[float, float]
            Intervalo de tempo (t_inicio, t_fim)
        t_eval : np.ndarray, optional
            Pontos de tempo específicos onde avaliar a solução
        method : str, default='RK45'
            Método de integração numérica

        Retorna
        -------
        dict
            Dicionário com 't', 'S', 'E', 'I', 'R'
        """
        S0, E0, I0, R0 = initial_conditions

        # Resolver sistema de ODEs
        solution = solve_ivp(
            fun=self.derivatives,
            t_span=t_span,
            y0=initial_conditions,
            t_eval=t_eval,
            method=method,
            dense_output=True
        )

        if not solution.success:
            raise RuntimeError(f"Integração falhou: {solution.message}")

        return {
            't': solution.t,
            'S': solution.y[0],
            'E': solution.y[1],
            'I': solution.y[2],
            'R': solution.y[3]
        }

    def compute_R0(self) -> float:
        """
        Calcula o número básico de reprodução R₀ = β / γ

        Note que R₀ não depende de σ (taxa de incubação).
        A taxa de incubação afeta a dinâmica temporal mas não
        o número médio de infecções secundárias.

        Interpretação:
        - R₀ > 1: A epidemia cresce
        - R₀ = 1: A epidemia é estável
        - R₀ < 1: A epidemia decresce

        Retorna
        -------
        float
            Número básico de reprodução
        """
        return self.beta / self.gamma

    def __repr__(self) -> str:
        R0 = self.compute_R0()
        incubation_period = 1 / self.sigma
        return (f"SEIRModel(beta={self.beta:.4f}, sigma={self.sigma:.4f}, "
                f"gamma={self.gamma:.4f}, N={self.N:.0f}, "
                f"R0={R0:.4f}, incubation_period={incubation_period:.1f} dias)")
