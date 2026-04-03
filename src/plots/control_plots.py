"""
Visualization of Optimal Control Results

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


# =============================================================================
# Generic plotting functions — English labels, consistent design
# Used by: optimal_control_pontryagin_seir2, _sir, _sidarthe
# =============================================================================

# Shared design constants
_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 10,
}

_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
           '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

_CTRL_COLOR   = '#1f77b4'    # blue — controlled
_NOCTRL_COLOR = '#aec7e8'    # light blue — uncontrolled
_U1_COLOR     = '#d62728'    # red
_U2_COLOR     = '#2ca02c'    # green


def _apply_style():
    """Apply shared matplotlib style."""
    plt.rcParams.update(_STYLE)


def _save(fig, path, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)


# --------------------------------------------------------------------------
# Control trajectories (shared by all models, max 2 subplots)
# --------------------------------------------------------------------------

def plot_control_trajectories_generic(
    solution: dict,
    save_dir: Path,
    model_name: str,
    dpi: int = 150,
    fmt: str = 'png',
    figsize=(12, 6)
):
    """
    Figure: optimal control trajectories u1(t) and u2(t).
    Single figure, 2 subplots (1x2 or 2x1).
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    t = solution['t']

    ax = axes[0]
    ax.plot(t, solution['u1'] * 100, color=_U1_COLOR, linewidth=2)
    ax.set_title('Social Distancing u₁(t)', fontweight='bold')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Lockdown intensity (%)')
    ax.set_ylim([0, 105])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))

    ax = axes[1]
    ax.plot(t, solution['u2'], color=_U2_COLOR, linewidth=2)
    ax.set_title('Vaccination u₂(t)', fontweight='bold')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Vaccination rate (persons/day)')
    ax.set_ylim(bottom=0)

    fig.suptitle(f'{model_name} — Optimal Control Trajectories',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    _save(fig, save_dir / f'{model_name.lower()}_control_trajectories.{fmt}', dpi)


# --------------------------------------------------------------------------
# Compartments (up to 4 per figure — split if needed)
# --------------------------------------------------------------------------

def plot_compartments_generic(
    solution: dict,
    solution_nc: dict,
    state_keys: list,
    state_labels: dict,
    save_dir: Path,
    model_name: str,
    dpi: int = 150,
    fmt: str = 'png',
    figsize=(12, 8)
):
    """
    Plot compartment trajectories (controlled vs uncontrolled).
    Max 4 compartments per figure. Splits into multiple figures if needed.
    """
    _apply_style()

    # Chunk into groups of 4
    chunks = [state_keys[i:i + 4] for i in range(0, len(state_keys), 4)]

    for fig_idx, chunk in enumerate(chunks):
        n = len(chunk)
        ncols = 2 if n > 1 else 1
        nrows = (n + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 squeeze=False)

        for idx, key in enumerate(chunk):
            ax = axes[idx // 2][idx % 2]
            label = state_labels.get(key, key)

            ax.plot(solution['t'], solution[key],
                    color=_CTRL_COLOR, linewidth=2, label='Optimal control')

            if key in solution_nc:
                ax.plot(solution_nc['t'], solution_nc[key],
                        color=_NOCTRL_COLOR, linewidth=2, linestyle='--',
                        alpha=0.8, label='No control')

            ax.set_title(label, fontweight='bold')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Population')
            ax.legend()
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f'{v/1e6:.1f}M' if abs(v) >= 1e6 else
                                  (f'{v/1e3:.1f}k' if abs(v) >= 1e3 else f'{v:.0f}'))
            )

        # Hide empty subplots
        for idx in range(len(chunk), nrows * ncols):
            axes[idx // 2][idx % 2].set_visible(False)

        suffix = f'_part{fig_idx + 1}' if len(chunks) > 1 else ''
        fig.suptitle(f'{model_name} — Compartments{suffix}',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        _save(fig,
              save_dir / f'{model_name.lower()}_compartments{suffix}.{fmt}',
              dpi)


# --------------------------------------------------------------------------
# Co-states (up to 4 per figure — split if needed)
# --------------------------------------------------------------------------

def plot_costates_generic(
    solution: dict,
    costate_keys: list,
    costate_labels: dict,
    save_dir: Path,
    model_name: str,
    dpi: int = 150,
    fmt: str = 'png',
    figsize=(12, 8)
):
    """
    Plot co-state (adjoint variable) trajectories.
    Max 4 per figure. Splits if needed.
    """
    _apply_style()

    chunks = [costate_keys[i:i + 4] for i in range(0, len(costate_keys), 4)]

    for fig_idx, chunk in enumerate(chunks):
        n = len(chunk)
        ncols = 2 if n > 1 else 1
        nrows = (n + 1) // 2
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 squeeze=False)

        for idx, key in enumerate(chunk):
            ax = axes[idx // 2][idx % 2]
            color = _COLORS[idx % len(_COLORS)]
            label = costate_labels.get(key, key)

            ax.plot(solution['t'], solution[key],
                    color=color, linewidth=2)
            ax.axhline(y=0, color='k', linestyle=':', alpha=0.4)
            ax.set_title(label, fontweight='bold')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Co-state value (R$)')

        for idx in range(len(chunk), nrows * ncols):
            axes[idx // 2][idx % 2].set_visible(False)

        suffix = f'_part{fig_idx + 1}' if len(chunks) > 1 else ''
        fig.suptitle(f'{model_name} — Co-states (Adjoint Variables){suffix}',
                     fontsize=14, fontweight='bold')
        fig.tight_layout()
        _save(fig,
              save_dir / f'{model_name.lower()}_costates{suffix}.{fmt}',
              dpi)
