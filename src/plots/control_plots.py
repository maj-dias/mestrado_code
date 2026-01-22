"""
Visualizacao de Resultados de Controle Otimo

Este modulo contem funcoes para visualizar resultados de controle otimo:
- Trajetorias de controle u1(t), u2(t)
- Estados controlados vs nao-controlados
- Co-estados (variaveis adjuntas)
- Analise de custo
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict


def plot_pontryagin_results(
    solution: Dict,
    solution_no_control: Dict,
    save_dir: Path,
    config: Dict
):
    """
    Gera graficos completos dos resultados de controle otimo Pontryagin

    Parametros
    ----------
    solution : dict
        Solucao do controle otimo (com controle)
    solution_no_control : dict
        Solucao sem controle (baseline)
    save_dir : Path
        Diretorio para salvar figuras
    config : dict
        Configuracao com parametros de plotagem
    """
    # Criar diretorio se nao existir
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Configuracoes de figuras
    fig_config = config['output']['figures']
    figsize = tuple(fig_config['figsize'])
    dpi = fig_config['dpi']
    fmt = fig_config['format']

    # Figura 1: Trajetorias de Controle
    plot_control_trajectories(solution, save_dir, figsize, dpi, fmt)

    # Figura 2: Compartimentos Controlados vs Nao-Controlados
    plot_compartments_comparison(solution, solution_no_control, save_dir, figsize, dpi, fmt)

    # Figura 3: Co-estados (variaveis adjuntas)
    plot_costates(solution, save_dir, figsize, dpi, fmt)

    print(f"   Total: 3 figuras geradas")


def plot_control_trajectories(solution, save_dir, figsize, dpi, fmt):
    """Plota trajetorias de controle u1(t) e u2(t)"""
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    t = solution['t']
    u1 = solution['u1']
    u2 = solution['u2']

    # u1: Lockdown
    axes[0].plot(t, u1 * 100, 'b-', linewidth=2)
    axes[0].set_ylabel('Lockdown u1 (%)', fontsize=12)
    axes[0].set_title('Trajetorias Otimas de Controle', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 100])

    # u2: Vacinacao
    axes[1].plot(t, u2, 'r-', linewidth=2)
    axes[1].set_xlabel('Tempo (dias)', fontsize=12)
    axes[1].set_ylabel('Vacinacao u2 (pessoas/dia)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, None])

    plt.tight_layout()
    plt.savefig(save_dir / f'seir_control_trajectories.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_compartments_comparison(solution, solution_no_control, save_dir, figsize, dpi, fmt):
    """Plota compartimentos SEIR: controlado vs nao-controlado"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    t = solution['t']

    compartments = ['S', 'E', 'I', 'R']
    titles = ['Suscetiveis', 'Expostos', 'Infectados', 'Removidos']
    colors_ctrl = ['blue', 'orange', 'red', 'green']
    colors_no_ctrl = ['lightblue', 'peachpuff', 'lightcoral', 'lightgreen']

    for idx, (comp, title, color_ctrl, color_no) in enumerate(zip(compartments, titles, colors_ctrl, colors_no_ctrl)):
        ax = axes[idx // 2, idx % 2]

        # Com controle
        ax.plot(t, solution[comp], color=color_ctrl, linewidth=2, label='Com controle')

        # Sem controle
        ax.plot(t, solution_no_control[comp], color=color_no, linewidth=2,
                linestyle='--', label='Sem controle', alpha=0.7)

        ax.set_xlabel('Tempo (dias)', fontsize=10)
        ax.set_ylabel(f'{title}', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle('Compartimentos SEIR: Controlado vs Nao-Controlado',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / f'seir_compartments_comparison.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_costates(solution, save_dir, figsize, dpi, fmt):
    """Plota co-estados (variaveis adjuntas) lambda(t)"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    t = solution['t']

    costates = ['lam_S', 'lam_E', 'lam_I', 'lam_R']
    titles = ['lambda_S', 'lambda_E', 'lambda_I', 'lambda_R']
    colors = ['blue', 'orange', 'red', 'green']

    for idx, (costate, title, color) in enumerate(zip(costates, titles, colors)):
        ax = axes[idx // 2, idx % 2]

        ax.plot(t, solution[costate], color=color, linewidth=2)
        ax.set_xlabel('Tempo (dias)', fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(f'Co-estado {title}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)

    plt.suptitle('Co-estados (Variaveis Adjuntas)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_dir / f'seir_costates.{fmt}', dpi=dpi, bbox_inches='tight')
    plt.close()
