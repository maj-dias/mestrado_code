"""
Funções de Visualização para Identificação de Parâmetros

Este módulo contém funções para gerar gráficos e relatórios visuais
dos resultados da identificação de parâmetros do modelo SIR.
"""

import matplotlib
# Usar backend não-interativo (Agg) para evitar problemas com Tcl/Tk
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Optional, Dict
import pandas as pd


def plot_sir_fit(
    time_data: np.ndarray,
    I_data: np.ndarray,
    R_data: np.ndarray,
    I_model: np.ndarray,
    R_model: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[Path] = None
):
    """
    Plota dados observados vs modelo ajustado

    Cria 2 subplots mostrando I(t) e R(t) com dados e predições do modelo

    Parâmetros
    ----------
    time_data : np.ndarray
        Tempos (dias)
    I_data : np.ndarray
        Dados observados de infectados
    R_data : np.ndarray
        Dados observados de removidos
    I_model : np.ndarray
        Predição do modelo para infectados
    R_model : np.ndarray
        Predição do modelo para removidos
    dates : pd.DatetimeIndex, optional
        Datas correspondentes aos tempos
    save_path : Path, optional
        Caminho para salvar a figura
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    x_axis = dates if dates is not None else time_data

    # Plot I(t)
    ax1.scatter(x_axis, I_data, label='Dados Observados', alpha=0.5, s=20, color='steelblue')
    ax1.plot(x_axis, I_model, label='Modelo SIR', color='red', linewidth=2.5)
    ax1.set_ylabel('Infectados (I)', fontsize=12, fontweight='bold')
    ax1.set_title('Ajuste do Modelo SIR - Infectados', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.ticklabel_format(style='plain', axis='y')

    # Plot R(t)
    ax2.scatter(x_axis, R_data, label='Dados Observados', alpha=0.5, s=20, color='darkgreen')
    ax2.plot(x_axis, R_model, label='Modelo SIR', color='orange', linewidth=2.5)
    ax2.set_ylabel('Removidos (R)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Data' if dates is not None else 'Dias', fontsize=12, fontweight='bold')
    ax2.set_title('Ajuste do Modelo SIR - Removidos (Recuperados + Óbitos)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.ticklabel_format(style='plain', axis='y')

    # Rotacionar labels de data se aplicável
    if dates is not None:
        ax1.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de ajuste salvo em: {save_path}")

    plt.close()


def plot_compartments_evolution(
    time_data: np.ndarray,
    S: np.ndarray,
    I: np.ndarray,
    R: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[Path] = None
):
    """
    Plota evolução de todos os compartimentos S, I, R

    Parâmetros
    ----------
    time_data : np.ndarray
        Tempos (dias)
    S, I, R : np.ndarray
        Compartimentos do modelo SIR
    dates : pd.DatetimeIndex, optional
        Datas correspondentes
    save_path : Path, optional
        Caminho para salvar a figura
    """
    plt.figure(figsize=(14, 7))

    x_axis = dates if dates is not None else time_data

    plt.plot(x_axis, S, label='Suscetíveis (S)', linewidth=2.5, color='dodgerblue')
    plt.plot(x_axis, I, label='Infectados (I)', linewidth=2.5, color='red')
    plt.plot(x_axis, R, label='Removidos (R)', linewidth=2.5, color='green')

    plt.xlabel('Data' if dates is not None else 'Dias', fontsize=12, fontweight='bold')
    plt.ylabel('População', fontsize=12, fontweight='bold')
    plt.title('Evolução dos Compartimentos SIR', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ticklabel_format(style='plain', axis='y')

    if dates is not None:
        plt.xticks(rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de compartimentos salvo em: {save_path}")

    plt.close()


def plot_residuals(
    time_data: np.ndarray,
    I_residuals: np.ndarray,
    R_residuals: np.ndarray,
    dates: Optional[pd.DatetimeIndex] = None,
    save_path: Optional[Path] = None
):
    """
    Análise de resíduos (dados - modelo)

    Cria 2x2 subplots mostrando:
    - Resíduos de I ao longo do tempo
    - Resíduos de R ao longo do tempo
    - Histograma dos resíduos de I
    - Histograma dos resíduos de R

    Parâmetros
    ----------
    time_data : np.ndarray
        Tempos (dias)
    I_residuals : np.ndarray
        Resíduos de infectados (dados - modelo)
    R_residuals : np.ndarray
        Resíduos de removidos (dados - modelo)
    dates : pd.DatetimeIndex, optional
        Datas correspondentes
    save_path : Path, optional
        Caminho para salvar a figura
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))

    x_axis = dates if dates is not None else time_data

    # Resíduos I ao longo do tempo
    axes[0, 0].scatter(x_axis, I_residuals, alpha=0.6, s=15, color='steelblue')
    axes[0, 0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Resíduos de I(t) ao Longo do Tempo', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Resíduo (Dados - Modelo)', fontsize=11)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    if dates is not None:
        axes[0, 0].tick_params(axis='x', rotation=45)

    # Resíduos R ao longo do tempo
    axes[0, 1].scatter(x_axis, R_residuals, alpha=0.6, s=15, color='darkgreen')
    axes[0, 1].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Resíduos de R(t) ao Longo do Tempo', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Resíduo (Dados - Modelo)', fontsize=11)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    if dates is not None:
        axes[0, 1].tick_params(axis='x', rotation=45)

    # Histograma resíduos I
    axes[1, 0].hist(I_residuals, bins=30, alpha=0.7, edgecolor='black', color='steelblue')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Distribuição dos Resíduos de I', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Resíduo', fontsize=11)
    axes[1, 0].set_ylabel('Frequência', fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--', axis='y')

    # Histograma resíduos R
    axes[1, 1].hist(R_residuals, bins=30, alpha=0.7, edgecolor='black', color='darkgreen')
    axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Distribuição dos Resíduos de R', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Resíduo', fontsize=11)
    axes[1, 1].set_ylabel('Frequência', fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Gráfico de resíduos salvo em: {save_path}")

    plt.close()


def generate_identification_report(
    results: Dict,
    data: Dict,
    save_dir: Path,
    config: Dict
):
    """
    Gera relatório completo com múltiplos gráficos

    Cria três gráficos principais:
    1. Ajuste do modelo aos dados (I e R)
    2. Evolução dos compartimentos S, I, R
    3. Análise de resíduos

    Parâmetros
    ----------
    results : dict
        Resultados da identificação (de identify_parameters)
    data : dict
        Dados SIR preparados (de prepare_sir_data)
    save_dir : Path
        Diretório onde salvar os gráficos
    config : dict
        Configurações de plotagem
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("GERANDO GRÁFICOS")
    print("="*60)

    # Configurar estilo se especificado
    if 'style' in config and config['style']:
        try:
            plt.style.use(config['style'])
        except:
            print(f"Aviso: Estilo '{config['style']}' não disponível, usando padrão")

    # Plot 1: Ajuste do modelo
    print("\n1. Gerando gráfico de ajuste do modelo...")
    plot_sir_fit(
        data['time'],
        data['I'],
        data['R'],
        results['prediction']['I'],
        results['prediction']['R'],
        dates=data.get('dates'),
        save_path=save_dir / 'sir_fit.png'
    )

    # Plot 2: Evolução dos compartimentos
    print("2. Gerando gráfico de evolução dos compartimentos...")
    plot_compartments_evolution(
        data['time'],
        results['prediction']['S'],
        results['prediction']['I'],
        results['prediction']['R'],
        dates=data.get('dates'),
        save_path=save_dir / 'compartments.png'
    )

    # Plot 3: Análise de resíduos
    print("3. Gerando gráfico de análise de resíduos...")
    I_residuals = data['I'] - results['prediction']['I']
    R_residuals = data['R'] - results['prediction']['R']

    plot_residuals(
        data['time'],
        I_residuals,
        R_residuals,
        dates=data.get('dates'),
        save_path=save_dir / 'residuals.png'
    )

    print(f"\nTodos os gráficos salvos em: {save_dir}")
    print("="*60)
