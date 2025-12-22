"""
Modelo SIR (Susceptible-Infected-Recovered)

Este módulo implementa o modelo epidemiológico compartimental SIR clássico.

O modelo SIR descreve a dinâmica de uma epidemia através de três compartimentos:
- S (Susceptibles): Indivíduos suscetíveis à infecção
- I (Infected): Indivíduos atualmente infectados
- R (Recovered): Indivíduos removidos (recuperados ou falecidos)

Equações diferenciais:
    dS/dt = -β * S * I / N
    dI/dt = β * S * I / N - γ * I
    dR/dt = γ * I

Onde:
- β (beta): Taxa de transmissão (contatos efetivos por dia)
- γ (gamma): Taxa de recuperação (1 / duração média da infecção)
- N: População total (constante, S + I + R = N)
"""

import numpy as np
from scipy.integrate import solve_ivp
from typing import List, Tuple, Dict, Optional


class SIRModel:
    """
    Modelo SIR para dinâmica epidemiológica

    Parâmetros
    ----------
    beta : float
        Taxa de transmissão (β > 0)
    gamma : float
        Taxa de recuperação (γ > 0)
    N : float
        População total (N > 0)
    """

    def __init__(self, beta: float, gamma: float, N: float):
        self.beta = beta
        self.gamma = gamma
        self.N = N
        self.validate_parameters()

    def validate_parameters(self):
        """Valida que todos os parâmetros são positivos"""
        if self.beta <= 0:
            raise ValueError(f"beta deve ser positivo, recebido: {self.beta}")
        if self.gamma <= 0:
            raise ValueError(f"gamma deve ser positivo, recebido: {self.gamma}")
        if self.N <= 0:
            raise ValueError(f"N (população) deve ser positivo, recebido: {self.N}")

    def derivatives(self, t: float, y: List[float]) -> np.ndarray:
        """
        Calcula as derivadas do sistema SIR

        Equações:
            dS/dt = -β * S * I / N
            dI/dt = β * S * I / N - γ * I
            dR/dt = γ * I
        """
        S, I, R = y

        infection_rate = self.beta * S * I / self.N

        dS = -infection_rate
        dI = infection_rate - self.gamma * I
        dR = self.gamma * I

        return np.array([dS, dI, dR])

    def simulate(
        self,
        initial_conditions: List[float],
        t_span: Tuple[float, float],
        t_eval: Optional[np.ndarray] = None,
        method: str = 'RK45'
    ) -> Dict[str, np.ndarray]:
        """
        Simula a evolução temporal do modelo SIR

        Parâmetros
        ----------
        initial_conditions : List[float]
            Condições iniciais [S0, I0, R0]
        t_span : Tuple[float, float]
            Intervalo de tempo (t_inicio, t_fim)
        t_eval : np.ndarray, optional
            Pontos de tempo específicos onde avaliar a solução

        Retorna
        -------
        dict
            Dicionário com 't', 'S', 'I', 'R'
        """
        S0, I0, R0 = initial_conditions

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
            'I': solution.y[1],
            'R': solution.y[2]
        }

    def compute_R0(self) -> float:
        """
        Calcula o número básico de reprodução R₀ = β / γ

        Interpretação:
        - R₀ > 1: A epidemia cresce
        - R₀ = 1: A epidemia é estável
        - R₀ < 1: A epidemia decresce
        """
        return self.beta / self.gamma

    def __repr__(self) -> str:
        R0 = self.compute_R0()
        return f"SIRModel(beta={self.beta:.4f}, gamma={self.gamma:.4f}, N={self.N:.0f}, R0={R0:.4f})"
