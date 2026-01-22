#!/usr/bin/env python3
"""
Controle Otimo SEIR usando Pontryagin Maximum Principle

Este script resolve o problema de controle otimo para o modelo SEIR COVID-19:
1. Carrega parametros identificados (beta, sigma, gamma)
2. Define problema de controle otimo (custos, limites)
3. Resolve usando Pontryagin Maximum Principle (shooting method)
4. Gera trajetorias otimas de controle u1*(t), u2*(t)
5. Simula sistema controlado S(t), E(t), I(t), R(t)
6. Salva resultados e visualiza

Uso:
    python scripts/optimal_control_pontryagin_seir.py
"""

import yaml
import json
import numpy as np
from pathlib import Path
import sys

# Adicionar diretorio raiz ao path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.seir_controlled import SEIRControlledModel
from src.control.pontryagin import PontryaginSolver
from src.control.cost_functions import QuadraticCost


def load_config(config_path='config/optimal_control_pontryagin_seir.yaml'):
    """Carrega configuracao do arquivo YAML"""
    config_file = ROOT_DIR / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Arquivo de configuracao nao encontrado: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_identified_parameters(params_file):
    """Carrega parametros identificados do modelo SEIR"""
    params_path = ROOT_DIR / params_file

    if not params_path.exists():
        raise FileNotFoundError(f"Arquivo de parametros nao encontrado: {params_path}")

    with open(params_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(filepath, data):
    """Salva dados em arquivo JSON"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    """
    Funcao principal de execucao

    Orquestra todo o workflow de controle otimo
    """
    try:
        # 1. Carregar configuracao
        config = load_config()

        print("="*70)
        print("CONTROLE OTIMO SEIR - COVID-19 BRASIL")
        print("Metodo: Pontryagin Maximum Principle (Shooting)")
        print("="*70)

        # 2. Carregar parametros identificados
        print("\n1. Carregando parametros identificados...")
        params = load_identified_parameters(config['model']['parameters_file'])

        beta = params['beta']
        sigma = params['sigma']
        gamma = params['gamma']
        N = params['config']['data']['population']

        print(f"   beta  = {beta:.6f}")
        print(f"   sigma = {sigma:.6f}")
        print(f"   gamma = {gamma:.6f}")
        print(f"   R0    = {beta/gamma:.4f}")
        print(f"   N     = {N:,.0f}")

        # 3. Condicoes iniciais
        print("\n2. Definindo condicoes iniciais...")
        if config['initial_conditions']['use_last_data_point']:
            # Usar ultimas condicoes dos dados identificados
            S0 = params['initial_conditions']['S0']
            E0 = params['initial_conditions']['E0']
            I0 = params['initial_conditions']['I0']
            R0 = params['initial_conditions']['R0']
        else:
            # Valores customizados (nao implementado)
            raise NotImplementedError("Condicoes iniciais customizadas nao implementadas")

        x0 = np.array([S0, E0, I0, R0])

        print(f"   S0 = {S0:>15,.0f} suscetivel")
        print(f"   E0 = {E0:>15,.0f} expostos")
        print(f"   I0 = {I0:>15,.0f} infectados")
        print(f"   R0 = {R0:>15,.0f} removidos")
        print(f"   Soma = {S0+E0+I0+R0:>15,.0f} (pop: {N:,.0f})")

        # 4. Criar modelo SEIR controlado
        print("\n3. Criando modelo SEIR controlado...")
        model = SEIRControlledModel(beta, sigma, gamma, N)
        print(f"   {model}")

        # 5. Definir funcao de custo
        print("\n4. Definindo funcao de custo...")
        w1 = float(config['cost']['weights']['w1_infections'])
        w2 = float(config['cost']['weights']['w2_lockdown'])
        w3 = float(config['cost']['weights']['w3_vaccination'])
        wf = float(config['cost']['weights']['wf_terminal'])

        cost = QuadraticCost(w1, w2, w3, wf)
        print(f"   {cost}")

        # 6. Limites de controle
        print("\n5. Definindo limites de controle...")
        u_bounds = {
            'u1': (config['control']['u1_lockdown']['min'],
                   config['control']['u1_lockdown']['max']),
            'u2': (config['control']['u2_vaccination']['min'],
                   config['control']['u2_vaccination']['max'])
        }

        print(f"   u1 (lockdown): [{u_bounds['u1'][0]:.2f}, {u_bounds['u1'][1]:.2f}]")
        print(f"   u2 (vacinacao): [{u_bounds['u2'][0]:.0f}, {u_bounds['u2'][1]:.0f}] pessoas/dia")

        # 7. Horizonte de controle
        T = config['cost']['horizon_days']
        print(f"\n6. Horizonte de controle: T = {T} dias")

        # 8. Criar solver Pontryagin
        print("\n7. Criando solver Pontryagin...")
        solver = PontryaginSolver(model, cost, u_bounds)

        # 9. Resolver problema de controle otimo
        print("\n8. Resolvendo problema de controle otimo...")
        solution = solver.solve(
            x0=x0,
            T=T,
            n_points=config['solver']['n_time_points'],
            tolerance=config['solver']['tolerance'],
            max_iterations=config['solver']['max_iterations']
        )

        # 10. Exibir resultados
        print("\n" + "="*70)
        print("RESULTADOS DO CONTROLE OTIMO")
        print("="*70)

        print(f"\nStatus: {'Sucesso' if solution['success'] else 'Falha'}")
        print(f"Custo total J: {solution['cost']:.4e}")
        print(f"Iteracoes: {solution['iterations']}")

        # Metricas de controle
        u1_mean = np.mean(solution['u1'])
        u2_mean = np.mean(solution['u2'])
        I_max = np.max(solution['I'])
        I_final = solution['I'][-1]

        print(f"\nMetricas de controle:")
        print(f"   Lockdown medio: {u1_mean*100:.1f}%")
        print(f"   Vacinacao media: {u2_mean:,.0f} pessoas/dia")
        print(f"   Pico de infeccoes: {I_max:,.0f}")
        print(f"   Infeccoes finais I(T): {I_final:,.0f}")

        # 11. Simular cenario sem controle para comparacao
        print("\n9. Simulando cenario sem controle para comparacao...")
        def zero_control(t):
            return np.array([0.0, 0.0])

        solution_no_control = model.simulate_with_control(
            x0, zero_control, (0, T), solution['t']
        )

        I_max_no_control = np.max(solution_no_control['I'])
        reduction = (I_max_no_control - I_max) / I_max_no_control * 100

        print(f"   Pico sem controle: {I_max_no_control:,.0f}")
        print(f"   Pico com controle: {I_max:,.0f}")
        print(f"   Reducao no pico: {reduction:.1f}%")

        # 12. Salvar resultados
        results_dir = ROOT_DIR / config['output']['results_dir']

        if config['output']['save_trajectories']:
            print("\n10. Salvando trajetorias de controle...")
            trajectories_file = results_dir / 'seir_optimal_trajectories.json'

            trajectories_data = {
                'time': solution['t'].tolist(),
                'u1_lockdown': solution['u1'].tolist(),
                'u2_vaccination': solution['u2'].tolist(),
                'total_cost': float(solution['cost']),
                'config': config,
                'model_parameters': {
                    'beta': float(beta),
                    'sigma': float(sigma),
                    'gamma': float(gamma),
                    'R0': float(beta/gamma)
                },
                'metrics': {
                    'lockdown_mean': float(u1_mean),
                    'vaccination_mean': float(u2_mean),
                    'I_peak_controlled': float(I_max),
                    'I_peak_no_control': float(I_max_no_control),
                    'peak_reduction_percent': float(reduction),
                    'I_final': float(I_final)
                }
            }

            save_json(trajectories_file, trajectories_data)
            print(f"   Salvo em: {trajectories_file}")

        if config['output']['save_states']:
            print("\n11. Salvando estados controlados...")
            states_file = results_dir / 'seir_controlled_states.json'

            states_data = {
                'time': solution['t'].tolist(),
                'S': solution['S'].tolist(),
                'E': solution['E'].tolist(),
                'I': solution['I'].tolist(),
                'R': solution['R'].tolist(),
                'costates': {
                    'lambda_S': solution['lam_S'].tolist(),
                    'lambda_E': solution['lam_E'].tolist(),
                    'lambda_I': solution['lam_I'].tolist(),
                    'lambda_R': solution['lam_R'].tolist()
                },
                'comparison_no_control': {
                    'S': solution_no_control['S'].tolist(),
                    'E': solution_no_control['E'].tolist(),
                    'I': solution_no_control['I'].tolist(),
                    'R': solution_no_control['R'].tolist()
                }
            }

            save_json(states_file, states_data)
            print(f"   Salvo em: {states_file}")

        # 13. Gerar graficos (se implementado)
        if config['output']['save_figures']:
            print("\n12. Gerando graficos...")
            try:
                from src.plots.control_plots import plot_pontryagin_results
                figures_dir = Path(config['output']['figures'].get('save_dir', results_dir / 'figures'))
                plot_pontryagin_results(solution, solution_no_control, figures_dir, config)
                print(f"   Graficos salvos em: {figures_dir}")
            except ImportError:
                print("   Aviso: Modulo de plots nao encontrado. Pulando graficos.")

        print("\n" + "="*70)
        print("CONTROLE OTIMO CONCLUIDO COM SUCESSO!")
        print("="*70)

        return 0

    except Exception as e:
        print(f"\nERRO: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
