#!/usr/bin/env python3
"""
Identificação de Parâmetros do Modelo SEIR usando Dados COVID-19 do Brasil

Este script orquestra o workflow completo de identificação:
1. Carregar dados COVID-19 do Brasil
2. Preparar compartimentos SEIR
3. Estimar parâmetros β, σ e γ usando mínimos quadrados
4. Gerar gráficos de diagnóstico
5. Salvar resultados

Uso:
    python scripts/identification_seir.py
"""

import yaml
import logging
import json
from pathlib import Path
import sys

# Adicionar diretório raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.data_loader import load_covid_data_brazil, prepare_seir_data
from src.identification.least_squares import identify_parameters_seir
from src.plots.identification import generate_identification_report


def load_config(config_path='config/seir.yaml'):
    """
    Carrega configuração do arquivo YAML

    Parâmetros
    ----------
    config_path : str
        Caminho para o arquivo de configuração

    Retorna
    -------
    dict
        Dicionário com configurações
    """
    config_file = ROOT_DIR / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_logging(config):
    """
    Configura sistema de logging

    Parâmetros
    ----------
    config : dict
        Configurações (deve conter seção 'logging')

    Retorna
    -------
    logging.Logger
        Logger configurado
    """
    logging.basicConfig(
        level=config['logging']['level'],
        format=config['logging']['format']
    )
    return logging.getLogger(__name__)


def main():
    """
    Função principal de execução

    Orquestra todo o workflow de identificação de parâmetros SEIR
    """
    try:
        # 1. Carregar configuração
        config = load_config()
        logger = setup_logging(config)

        logger.info("=" * 70)
        logger.info("IDENTIFICAÇÃO DE PARÂMETROS SEIR - COVID-19 BRASIL")
        logger.info("=" * 70)

        # 2. Carregar dados COVID-19
        logger.info("\n📂 Carregando dados COVID-19...")
        data_dir = ROOT_DIR / config['data']['directory']
        start_date = config['data']['start_date']
        end_date = config['data']['end_date']
        N = config['data']['population']

        df = load_covid_data_brazil(str(data_dir), start_date, end_date)
        logger.info(f"✓ Dados carregados: {len(df)} registros")

        # 3. Preparar compartimentos SEIR
        logger.info("\n🔬 Preparando compartimentos SEIR...")
        seir_data = prepare_seir_data(
            df, N,
            smooth=config['data']['use_smoothing'],
            window=config['data']['smooth_window']
        )

        logger.info(f"✓ Período: {seir_data['dates'][0].date()} a {seir_data['dates'][-1].date()}")
        logger.info(f"✓ Número de dias: {len(seir_data['time'])}")
        logger.info(f"\nCondicoes iniciais:")
        logger.info(f"   S₀ = {seir_data['S'][0]:>15,.0f} suscetíveis")
        logger.info(f"   E₀ = {seir_data['E'][0]:>15,.0f} expostos")
        logger.info(f"   I₀ = {seir_data['I'][0]:>15,.0f} infectados")
        logger.info(f"   R₀ = {seir_data['R'][0]:>15,.0f} removidos")
        logger.info(f"   N  = {seir_data['population']:>15,.0f} população total")

        # 4. Executar identificação
        logger.info("\nIniciando identificacao de parametros...")
        logger.info(f"   Método: {config['identification']['method']}")
        logger.info(f"   Otimizador: {config['identification']['optimizer']['algorithm']}")
        logger.info(f"   Chute inicial: β={config['identification']['initial_guess']['beta']:.4f}, "
                   f"σ={config['identification']['initial_guess']['sigma']:.4f}, "
                   f"γ={config['identification']['initial_guess']['gamma']:.4f}")

        results = identify_parameters_seir(
            time_data=seir_data['time'],
            E_data=seir_data['E'],
            I_data=seir_data['I'],
            R_data=seir_data['R'],
            N=N,
            initial_guess=config['identification']['initial_guess'],
            bounds=config['identification']['bounds'],
            method=config['identification']['optimizer']['algorithm']
        )

        # 5. Exibir resultados
        logger.info("\n" + "=" * 70)
        logger.info("📈 RESULTADOS DA IDENTIFICAÇÃO")
        logger.info("=" * 70)
        logger.info(f"Status: {'✓ SUCESSO' if results['success'] else '✗ FALHA'}")

        logger.info(f"\nParametros estimados:")
        logger.info(f"   β (taxa de transmissão)   = {results['beta']:.6f}")
        logger.info(f"   σ (taxa de incubação)     = {results['sigma']:.6f}")
        logger.info(f"   γ (taxa de recuperação)   = {results['gamma']:.6f}")
        logger.info(f"   R₀ (número de reprodução)  = {results['R0']:.4f}")

        # Interpretação de R₀
        if results['R0'] > 1:
            interpretacao = "ALERTA: Epidemia em crescimento (R0 > 1)"
        elif results['R0'] < 1:
            interpretacao = "✓ Epidemia em declínio (R₀ < 1)"
        else:
            interpretacao = "→ Epidemia estável (R₀ ≈ 1)"
        logger.info(f"   {interpretacao}")

        # Duração média da infecção
        duracao_infeccao = 1 / results['gamma']
        logger.info(f"   Duração média da infecção: {duracao_infeccao:.1f} dias")

        # Período de incubação
        periodo_incubacao = 1 / results['sigma']
        logger.info(f"   Período de incubação: {periodo_incubacao:.1f} dias")

        logger.info(f"\nMetricas de ajuste:")
        logger.info(f"   R² (coef. determinação) = {results['metrics']['r2']:.4f}")
        logger.info(f"   RMSE                    = {results['metrics']['rmse']:,.2f}")
        logger.info(f"   RSS                     = {results['metrics']['rss']:.2e}")

        logger.info(f"\n⚙️  Detalhes da otimização:")
        logger.info(f"   Iterações: {results['iterations']}")
        logger.info(f"   Custo final: {results['cost']:.2e}")
        logger.info(f"   Mensagem: {results['message']}")

        # 6. Salvar resultados
        results_dir = ROOT_DIR / config['output']['results_dir']

        if config['output']['save_parameters']:
            logger.info("\n💾 Salvando parâmetros...")
            params_dir = results_dir / 'parameters'
            params_dir.mkdir(parents=True, exist_ok=True)
            params_file = params_dir / 'seir_params.json'

            # Preparar dados para JSON
            results_json = {
                'beta': float(results['beta']),
                'sigma': float(results['sigma']),
                'gamma': float(results['gamma']),
                'R0': float(results['R0']),
                'infectious_period_days': float(1/results['gamma']),
                'incubation_period_days': float(1/results['sigma']),
                'cost': float(results['cost']),
                'success': results['success'],
                'message': results['message'],
                'iterations': int(results['iterations']),
                'metrics': {k: float(v) for k, v in results['metrics'].items()},
                'config': config,
                'data_period': {
                    'start': str(seir_data['dates'][0].date()),
                    'end': str(seir_data['dates'][-1].date()),
                    'days': len(seir_data['time'])
                },
                'initial_conditions': {
                    'S0': float(seir_data['S'][0]),
                    'E0': float(seir_data['E'][0]),
                    'I0': float(seir_data['I'][0]),
                    'R0': float(seir_data['R'][0])
                }
            }

            with open(params_file, 'w', encoding='utf-8') as f:
                json.dump(results_json, f, indent=2, ensure_ascii=False)

            logger.info(f"   ✓ Parâmetros salvos em: {params_file}")

        # 7. Gerar gráficos
        if config['output']['save_figures']:
            logger.info("\nGerando graficos...")
            figures_dir = Path(config['output']['figures'].get('save_dir', results_dir / 'figures'))

            generate_identification_report(
                results=results,
                data=seir_data,
                save_dir=figures_dir,
                config=config['output']['figures']
            )

            logger.info(f"   Graficos salvos em: {figures_dir}")

        logger.info("\n" + "=" * 70)
        logger.info("IDENTIFICACAO CONCLUIDA COM SUCESSO!")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"\nERRO: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())
