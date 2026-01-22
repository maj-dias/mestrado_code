"""
Funções de Custo para Controle Ótimo

Este módulo define funções de custo para problemas de controle ótimo:
- Custo quadrático (running cost)
- Custo terminal
- Condições terminais para co-estados
"""

import numpy as np


class QuadraticCost:
    """
    Função de custo quadrática para controle ótimo

    Custo instantâneo (running cost):
        L(x, u, t) = w1·I(t) + w2·u1²(t) + w3·u2²(t)

    Custo terminal:
        Φ(x(T)) = wf·I(T)

    Custo total:
        J = ∫₀ᵀ L(x, u, t) dt + Φ(x(T))

    Parâmetros
    ----------
    w1 : float
        Peso do custo de infecções (pessoas infectadas)
    w2 : float
        Peso do custo de lockdown (impacto econômico)
    w3 : float
        Peso do custo de vacinação (custo logístico)
    wf : float
        Peso do custo terminal (penalidade por infecções no final)
    """

    def __init__(self, w1: float, w2: float, w3: float, wf: float):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.wf = wf

    def running_cost(self, x: np.ndarray, u: np.ndarray, t: float) -> float:
        """
        Calcula o custo instantâneo L(x, u, t)

        Parâmetros
        ----------
        x : np.ndarray
            Vetor de estado [S, E, I, R]
        u : np.ndarray
            Vetor de controle [u1, u2]
        t : float
            Tempo atual

        Retorna
        -------
        float
            Custo instantâneo
        """
        S, E, I, R = x
        u1, u2 = u

        # Custo = infecções + lockdown² + vacinação²
        cost = self.w1 * I + self.w2 * u1**2 + self.w3 * u2**2

        return cost

    def terminal_cost(self, x: np.ndarray) -> float:
        """
        Calcula o custo terminal Φ(x(T))

        Parâmetros
        ----------
        x : np.ndarray
            Vetor de estado final [S(T), E(T), I(T), R(T)]

        Retorna
        -------
        float
            Custo terminal
        """
        S, E, I, R = x
        return self.wf * I

    def terminal_costates(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula as condições terminais dos co-estados λ(T) = ∂Φ/∂x

        Para Φ = wf·I(T):
            ∂Φ/∂S = 0
            ∂Φ/∂E = 0
            ∂Φ/∂I = wf
            ∂Φ/∂R = 0

        Parâmetros
        ----------
        x : np.ndarray
            Vetor de estado final [S(T), E(T), I(T), R(T)]

        Retorna
        -------
        np.ndarray
            Condições terminais [λS(T), λE(T), λI(T), λR(T)]
        """
        return np.array([0.0, 0.0, self.wf, 0.0])

    def __repr__(self) -> str:
        return (f"QuadraticCost(w1={self.w1:.2e}, w2={self.w2:.2e}, "
                f"w3={self.w3:.2e}, wf={self.wf:.2e})")
