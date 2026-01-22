#!/usr/bin/env python3
"""
Script de Identificação de Parâmetros - Modelo SIDARTHE

Este script identifica os 12 parâmetros do modelo SIDARTHE a partir de
dados reais de COVID-19 no Brasil usando otimização numérica.

Workflow:
1. Carrega dados de COVID-19 (casos, mortes, recuperados, ativos)
2. Carrega configuração (pesos, limites, etc.)
3. Executa identificação (Differential Evolution + L-BFGS-B)
4. Salva parâmetros identificados
5. Gera gráficos de ajuste
6. Calcula métricas de qualidade

Uso:
    python scripts/identification_sidarthe.py

Saída:
    - results/parameters/sidarthe_params.json
    - results/figures/identification/sidarthe_fit.png
"""

import yaml
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sem GUI
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.sidarthe import SIDARTHEModel
from src.identification.sidarthe_identifier import SIDARTHEIdentifier
from src.utils.data_loader import load_covid_data_brazil, prepare_sidarthe_data


def load_config(config_path='config/identification_sidarthe.yaml'):
    """Carrega configuração do arquivo YAML"""
    config_file = ROOT_DIR / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_data(config):
    """
    Carrega dados de COVID-19 usando data_loader

    Retorna
    -------
    pd.DataFrame
        DataFrame com dados filtrados e preparados para SIDARTHE
    """
    # Carregar dados brutos do Brasil
    df_raw = load_covid_data_brazil(
        data_dir=str(ROOT_DIR / 'data' / 'raw'),
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date']
    )

    # Preparar dados no formato SIDARTHE
    df_sidarthe = prepare_sidarthe_data(
        df=df_raw,
        smooth=True,
        window=7
    )

    return df_sidarthe


def save_json(filepath, data):
    """Salva dados em arquivo JSON"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def plot_fit(identifier, params, config, output_dir):
    """
    Gera gráfico comparando dados reais vs modelo ajustado

    Parâmetros
    ----------
    identifier : SIDARTHEIdentifier
        Objeto identificador com dados observados
    params : dict
        Parâmetros identificados
    config : dict
        Configuração
    output_dir : Path
        Diretório para salvar figura
    """
    # Simular com parâmetros identificados
    y0 = identifier._estimate_initial_conditions(params)
    model = SIDARTHEModel(params, identifier.N)

    sol = model.simulate(
        y0=y0,
        t_span=(0, len(identifier.data)),
        t_eval=identifier.t_obs
    )

    obs_sim = model.compute_observables(sol)

    # Criar figura
    fig_config = config['output']['figures']
    figsize = tuple(fig_config['figsize'])

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # 1. Casos confirmados
    axes[0, 0].plot(identifier.t_obs, identifier.cases_obs, 'o',
                    label='Dados reais', alpha=0.6, markersize=4)
    axes[0, 0].plot(identifier.t_obs, obs_sim['confirmed'], '-',
                    label='Modelo ajustado', linewidth=2, color='blue')
    axes[0, 0].set_title('Casos Confirmados', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Dias desde início')
    axes[0, 0].set_ylabel('Casos acumulados')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Mortes
    axes[0, 1].plot(identifier.t_obs, identifier.deaths_obs, 'o',
                    label='Dados reais', alpha=0.6, markersize=4, color='red')
    axes[0, 1].plot(identifier.t_obs, obs_sim['deaths'], '-',
                    label='Modelo ajustado', linewidth=2, color='darkred')
    axes[0, 1].set_title('Mortes Acumuladas', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Dias desde início')
    axes[0, 1].set_ylabel('Mortes')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Recuperados
    axes[1, 0].plot(identifier.t_obs, identifier.recovered_obs, 'o',
                    label='Dados reais', alpha=0.6, markersize=4, color='green')
    axes[1, 0].plot(identifier.t_obs, obs_sim['recovered'], '-',
                    label='Modelo ajustado', linewidth=2, color='darkgreen')
    axes[1, 0].set_title('Recuperados', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Dias desde início')
    axes[1, 0].set_ylabel('Recuperados acumulados')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Casos ativos
    axes[1, 1].plot(identifier.t_obs, identifier.active_obs, 'o',
                    label='Dados reais', alpha=0.6, markersize=4, color='orange')
    axes[1, 1].plot(identifier.t_obs, obs_sim['active'], '-',
                    label='Modelo ajustado', linewidth=2, color='darkorange')
    axes[1, 1].set_title('Casos Ativos', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Dias desde início')
    axes[1, 1].set_ylabel('Casos ativos')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Ajuste do Modelo SIDARTHE aos Dados de COVID-19',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Salvar
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'sidarthe_fit.{fig_config["format"]}'
    plt.savefig(output_file, dpi=fig_config['dpi'], bbox_inches='tight')
    plt.close()

    print(f"\n   Gráfico salvo: {output_file}")


def plot_compartments(identifier, params, config, output_dir):
    """
    Gera gráfico mostrando evolução de todos os 8 compartimentos

    Parâmetros
    ----------
    identifier : SIDARTHEIdentifier
        Objeto identificador
    params : dict
        Parâmetros identificados
    config : dict
        Configuração
    output_dir : Path
        Diretório para salvar figura
    """
    # Simular
    y0 = identifier._estimate_initial_conditions(params)
    model = SIDARTHEModel(params, identifier.N)

    sol = model.simulate(
        y0=y0,
        t_span=(0, len(identifier.data)),
        t_eval=identifier.t_obs
    )

    # Criar figura
    fig_config = config['output']['figures']
    figsize = tuple(fig_config['figsize'])

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()

    compartments = ['S', 'I', 'D', 'A', 'R', 'T', 'H', 'E']
    titles = ['Susceptible', 'Infected (não detect.)', 'Diagnosed (assint.)',
              'Ailing (sint. não detect.)', 'Recognized (sint. detect.)',
              'Threatened (graves)', 'Healed', 'Extinct']
    colors = ['blue', 'orange', 'cyan', 'yellow', 'magenta', 'red', 'green', 'black']

    for idx, (comp, title, color) in enumerate(zip(compartments, titles, colors)):
        axes[idx].plot(sol['t'], sol[comp], linewidth=2, color=color)
        axes[idx].set_title(title, fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Dias', fontsize=9)
        axes[idx].set_ylabel('População', fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

    plt.suptitle('Evolução dos 8 Compartimentos - Modelo SIDARTHE',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Salvar
    output_dir = Path(output_dir)
    output_file = output_dir / f'sidarthe_compartments.{fig_config["format"]}'
    plt.savefig(output_file, dpi=fig_config['dpi'], bbox_inches='tight')
    plt.close()

    print(f"   Gráfico salvo: {output_file}")


def main():
    """Função principal de execução"""
    try:
        print("="*70)
        print("IDENTIFICAÇÃO DE PARÂMETROS - MODELO SIDARTHE")
        print("Modelo: 8 compartimentos, 12 parâmetros")
        print("Método: Differential Evolution + L-BFGS-B")
        print("="*70)

        # 1. Carregar configuração
        print("\n1. Carregando configuração...")
        config = load_config()
        print(f"   Configuração: config/identification_sidarthe.yaml")
        print(f"   Modelo: {config['model']['name']}")
        print(f"   Referência: {config['model']['reference']}")

        # 2. Carregar dados
        print("\n2. Carregando dados de COVID-19...")
        data = load_data(config)

        print(f"   Fonte: data/raw/ (Ministério da Saúde)")
        print(f"   Período: {config['data']['start_date']} a {config['data']['end_date']}")
        print(f"   Total de dias: {len(data)}")
        print(f"   População: {config['data']['population']:,}")

        # Estatísticas dos dados
        print(f"\n   Estatísticas dos dados:")
        print(f"     Casos confirmados: {data['confirmed'].iloc[0]:,.0f} -> {data['confirmed'].iloc[-1]:,.0f}")
        print(f"     Mortes: {data['deaths'].iloc[0]:,.0f} -> {data['deaths'].iloc[-1]:,.0f}")
        print(f"     Recuperados: {data['recovered'].iloc[0]:,.0f} -> {data['recovered'].iloc[-1]:,.0f}")
        print(f"     Casos ativos (pico): {data['active'].max():,.0f}")

        # 3. Criar identificador
        print("\n3. Criando identificador...")
        identifier = SIDARTHEIdentifier(
            data=data,
            N=config['data']['population'],
            config=config
        )

        print(f"   Método: {config['optimization']['global_method']} -> {config['optimization']['local_method']}")
        print(f"   Parâmetros a identificar: 12")
        print(f"   Pesos: casos={config['optimization']['weights']['cases']}, "
              f"mortes={config['optimization']['weights']['deaths']}")

        # 4. Executar identificação
        print("\n4. Executando identificação...")
        print("   (Isto pode levar vários minutos...)")

        results = identifier.identify()

        # 5. Salvar resultados
        print("\n5. Salvando resultados...")
        output_dir = ROOT_DIR / config['output']['directory']
        output_file = output_dir / 'sidarthe_params.json'

        output_data = {
            'model': 'SIDARTHE',
            'description': config['model']['description'],
            'reference': config['model']['reference'],
            'parameters': results['parameters'],
            'metrics': results['metrics'],
            'optimization': results['optimization'],
            'config': config,
            'data_period': {
                'start': config['data']['start_date'],
                'end': config['data']['end_date'],
                'days': len(data)
            },
            'timestamp': datetime.now().isoformat()
        }

        save_json(output_file, output_data)
        print(f"   Parâmetros salvos: {output_file}")

        # 6. Gerar gráficos
        if config['output']['save_fit_plot']:
            print("\n6. Gerando gráficos...")
            figures_dir = ROOT_DIR / config['output']['figures']['save_dir']

            plot_fit(identifier, results['parameters'], config, figures_dir)
            plot_compartments(identifier, results['parameters'], config, figures_dir)

        # 7. Resumo final
        print("\n" + "="*70)
        print("IDENTIFICAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*70)

        print(f"\nArquivos gerados:")
        print(f"  - Parâmetros: {output_file}")
        if config['output']['save_fit_plot']:
            print(f"  - Gráficos: {figures_dir}/")

        print(f"\nPróximos passos:")
        print(f"  1. Validar ajuste visual nos gráficos")
        print(f"  2. Verificar métricas (R² > 0.95 é bom)")
        print(f"  3. Usar parâmetros em controle ótimo")

        return 0

    except Exception as e:
        print(f"\nERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
