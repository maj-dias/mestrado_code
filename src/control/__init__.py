"""
Módulo de Controle Ótimo

Este módulo contém implementações de controle ótimo para modelos epidemiológicos:
- Funções de custo (quadrático, terminal)
- Solver de Pontryagin (PMP - Princípio do Máximo de Pontryagin)
- Restrições e limites de controle
"""

from .cost_functions import (QuadraticCost, RealisticCostSEIR,
                             RealisticCostSIR, RealisticCostSIDARTHE)
from .pontryagin import PontryaginSolver, GenericPontryaginSolver

__all__ = ['QuadraticCost', 'RealisticCostSEIR', 'RealisticCostSIR',
           'RealisticCostSIDARTHE', 'PontryaginSolver',
           'GenericPontryaginSolver']
