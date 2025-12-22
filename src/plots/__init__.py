"""Funções de visualização"""
from .identification import (
    plot_sir_fit,
    plot_compartments_evolution,
    plot_residuals,
    generate_identification_report
)

__all__ = [
    'plot_sir_fit',
    'plot_compartments_evolution',
    'plot_residuals',
    'generate_identification_report'
]
