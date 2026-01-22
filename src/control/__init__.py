"""
Módulo de Controle Ótimo

Este módulo contém implementações de controle ótimo para modelos epidemiológicos:
- Funções de custo (quadrático, terminal)
- Solver de Pontryagin (PMP - Princípio do Máximo de Pontryagin)
- Restrições e limites de controle
"""

from .cost_functions import QuadraticCost
from .pontryagin import PontryaginSolver

__all__ = ['QuadraticCost', 'PontryaginSolver']
