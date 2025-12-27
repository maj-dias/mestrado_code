"""
Identificação de Parâmetros por Mínimos Quadrados

Este módulo implementa algoritmos de identificação de parâmetros para modelos
epidemiológicos usando otimização por mínimos quadrados não-lineares.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict
import sys
from pathlib import Path

# Adicionar src ao path para importações
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.sir import SIRModel


def cost_function(params: np.ndarray, time_data: np.ndarray,
                  I_data: np.ndarray, R_data: np.ndarray, N: float) -> float:
    """
    Função de custo: soma dos resíduos quadrados (RSS)

    Calcula a diferença entre os dados observados e a predição do modelo
    SIR para os parâmetros dados.

    Parâmetros
    ----------
    params : np.ndarray
        Parâmetros [beta, gamma]
    time_data : np.ndarray
        Vetor de tempos (dias)
    I_data : np.ndarray
        Dados observados de infectados
    R_data : np.ndarray
        Dados observados de removidos
    N : float
        População total

    Retorna
    -------
    float
        RSS (Residual Sum of Squares) - soma dos quadrados dos resíduos
    """
    try:
        beta, gamma = params

        # Validar parâmetros
        if beta <= 0 or gamma <= 0:
            return 1e10  # Penalidade grande para parâmetros inválidos

        # Condições iniciais dos dados
        S0 = N - I_data[0] - R_data[0]
        I0 = I_data[0]
        R0 = R_data[0]

        # Criar modelo com parâmetros atuais
        model = SIRModel(beta, gamma, N)

        # Simular modelo
        solution = model.simulate(
            initial_conditions=[S0, I0, R0],
            t_span=(time_data[0], time_data[-1]),
            t_eval=time_data
        )

        # Predições do modelo
        I_pred = solution['I']
        R_pred = solution['R']

        # Calcular resíduos
        residuals_I = I_data - I_pred
        residuals_R = R_data - R_pred

        # Soma dos quadrados dos resíduos
        rss = np.sum(residuals_I**2) + np.sum(residuals_R**2)

        return rss

    except Exception as e:
        # Em caso de erro na simulação, retornar custo muito alto
        print(f"Erro na avaliação da função de custo: {e}")
        return 1e10


def identify_parameters(
    time_data: np.ndarray,
    I_data: np.ndarray,
    R_data: np.ndarray,
    N: float,
    initial_guess: Dict[str, float],
    bounds: Dict[str, list],
    method: str = 'L-BFGS-B'
) -> Dict:
    """
    Identifica parâmetros β e γ do modelo SIR usando otimização

    Utiliza scipy.optimize.minimize para encontrar os parâmetros que
    minimizam a diferença entre o modelo e os dados observados.

    Parâmetros
    ----------
    time_data : np.ndarray
        Array de tempos (dias desde o início)
    I_data : np.ndarray
        Dados observados de infectados
    R_data : np.ndarray
        Dados observados de removidos
    N : float
        População total
    initial_guess : dict
        Chute inicial {'beta': valor, 'gamma': valor}
    bounds : dict
        Limites dos parâmetros {'beta': [min, max], 'gamma': [min, max]}
    method : str, default='L-BFGS-B'
        Algoritmo de otimização

    Retorna
    -------
    dict
        Dicionário com resultados:
        - 'beta': float - Taxa de transmissão estimada
        - 'gamma': float - Taxa de recuperação estimada
        - 'R0': float - Número básico de reprodução
        - 'cost': float - Valor final da função de custo
        - 'success': bool - Se a otimização foi bem-sucedida
        - 'message': str - Mensagem do otimizador
        - 'iterations': int - Número de iterações
        - 'metrics': dict - Métricas de qualidade do ajuste (R², RMSE)
        - 'prediction': dict - Predições do modelo ('S', 'I', 'R')
    """
    print("\n" + "="*60)
    print("OTIMIZAÇÃO DE PARÂMETROS")
    print("="*60)

    # Preparar chute inicial
    x0 = [initial_guess['beta'], initial_guess['gamma']]
    print(f"Chute inicial: β={x0[0]:.4f}, γ={x0[1]:.4f}")

    # Preparar limites
    param_bounds = [bounds['beta'], bounds['gamma']]
    print(f"Limites: β∈{bounds['beta']}, γ∈{bounds['gamma']}")

    # Callback para monitorar progresso
    iteration_count = [0]

    def callback(xk):
        iteration_count[0] += 1
        if iteration_count[0] % 10 == 0:
            cost = cost_function(xk, time_data, I_data, R_data, N)
            print(f"Iteração {iteration_count[0]}: β={xk[0]:.6f}, γ={xk[1]:.6f}, Custo={cost:.2e}")

    print(f"\nIniciando otimização com método {method}...")

    # Otimização
    result = minimize(
        fun=cost_function,
        x0=x0,
        args=(time_data, I_data, R_data, N),
        method=method,
        bounds=param_bounds,
        callback=callback,
        options={
            'maxiter': 1000,
            'ftol': 1e-9,
            'disp': False
        }
    )

    print(f"\nOtimizacao concluida: {result.message}")
    print(f"Total de iteracoes: {result.nit}")

    # Extrair parâmetros ótimos
    beta_opt, gamma_opt = result.x
    R0 = beta_opt / gamma_opt

    print(f"\nParâmetros estimados:")
    print(f"  β = {beta_opt:.6f}")
    print(f"  γ = {gamma_opt:.6f}")
    print(f"  R₀ = {R0:.4f}")

    # Simular com parâmetros ótimos para calcular métricas
    S0 = N - I_data[0] - R_data[0]
    model = SIRModel(beta_opt, gamma_opt, N)
    solution = model.simulate(
        initial_conditions=[S0, I_data[0], R_data[0]],
        t_span=(time_data[0], time_data[-1]),
        t_eval=time_data
    )

    I_pred = solution['I']
    R_pred = solution['R']
    S_pred = solution['S']

    # Calcular métricas de qualidade do ajuste

    # R² (coeficiente de determinação)
    ss_res = np.sum((I_data - I_pred)**2) + np.sum((R_data - R_pred)**2)
    ss_tot = np.sum((I_data - np.mean(I_data))**2) + np.sum((R_data - np.mean(R_data))**2)

    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = 0.0

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(ss_res / (2 * len(time_data)))

    # MAPE (Mean Absolute Percentage Error) para I
    mape_I = np.mean(np.abs((I_data - I_pred) / (I_data + 1))) * 100  # +1 para evitar divisão por zero

    print(f"\nMetricas de ajuste:")
    print(f"  R2 = {r2:.4f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAPE (I) = {mape_I:.2f}%")

    return {
        'beta': beta_opt,
        'gamma': gamma_opt,
        'R0': R0,
        'cost': result.fun,
        'success': result.success,
        'message': result.message,
        'iterations': result.nit,
        'metrics': {
            'r2': r2,
            'rmse': rmse,
            'mape_I': mape_I,
            'rss': result.fun
        },
        'prediction': {
            'S': S_pred,
            'I': I_pred,
            'R': R_pred
        }
    }


def cost_function_seir(params: np.ndarray, time_data: np.ndarray,
                       E_data: np.ndarray, I_data: np.ndarray,
                       R_data: np.ndarray, N: float) -> float:
    """
    Função de custo SEIR: soma dos resíduos quadrados (RSS)

    Calcula a diferença entre os dados observados e a predição do modelo
    SEIR para os parâmetros dados.

    Parâmetros
    ----------
    params : np.ndarray
        Parâmetros [beta, sigma, gamma]
    time_data : np.ndarray
        Vetor de tempos (dias)
    E_data : np.ndarray
        Dados observados de expostos (tipicamente zeros)
    I_data : np.ndarray
        Dados observados de infectados
    R_data : np.ndarray
        Dados observados de removidos
    N : float
        População total

    Retorna
    -------
    float
        RSS (Residual Sum of Squares) - soma dos quadrados dos resíduos
    """
    try:
        beta, sigma, gamma = params

        # Validar parâmetros
        if beta <= 0 or sigma <= 0 or gamma <= 0:
            return 1e10  # Penalidade grande para parâmetros inválidos

        # Condições iniciais dos dados (4 compartimentos)
        E0 = E_data[0]  # Tipicamente 0
        I0 = I_data[0]
        R0 = R_data[0]
        S0 = N - E0 - I0 - R0

        # Importar modelo SEIR (dentro da função para evitar dependência circular)
        from src.models.seir import SEIRModel

        # Criar modelo com parâmetros atuais
        model = SEIRModel(beta, sigma, gamma, N)

        # Simular modelo
        solution = model.simulate(
            initial_conditions=[S0, E0, I0, R0],
            t_span=(time_data[0], time_data[-1]),
            t_eval=time_data
        )

        # Predições do modelo
        I_pred = solution['I']
        R_pred = solution['R']

        # Calcular resíduos (fit apenas I e R, pois E_data é tipicamente zero)
        residuals_I = I_data - I_pred
        residuals_R = R_data - R_pred

        # Soma dos quadrados dos resíduos
        rss = np.sum(residuals_I**2) + np.sum(residuals_R**2)

        return rss

    except Exception as e:
        # Em caso de erro na simulação, retornar custo muito alto
        print(f"Erro na avaliação da função de custo SEIR: {e}")
        return 1e10


def identify_parameters_seir(
    time_data: np.ndarray,
    E_data: np.ndarray,
    I_data: np.ndarray,
    R_data: np.ndarray,
    N: float,
    initial_guess: Dict[str, float],
    bounds: Dict[str, list],
    method: str = 'L-BFGS-B'
) -> Dict:
    """
    Identifica parâmetros β, σ e γ do modelo SEIR usando otimização

    Utiliza scipy.optimize.minimize para encontrar os parâmetros que
    minimizam a diferença entre o modelo e os dados observados.

    Parâmetros
    ----------
    time_data : np.ndarray
        Array de tempos (dias desde o início)
    E_data : np.ndarray
        Dados observados de expostos
    I_data : np.ndarray
        Dados observados de infectados
    R_data : np.ndarray
        Dados observados de removidos
    N : float
        População total
    initial_guess : dict
        Chute inicial {'beta': valor, 'sigma': valor, 'gamma': valor}
    bounds : dict
        Limites dos parâmetros {'beta': [min, max], 'sigma': [min, max], 'gamma': [min, max]}
    method : str, default='L-BFGS-B'
        Algoritmo de otimização

    Retorna
    -------
    dict
        Dicionário com resultados:
        - 'beta': float - Taxa de transmissão estimada
        - 'sigma': float - Taxa de incubação estimada
        - 'gamma': float - Taxa de recuperação estimada
        - 'R0': float - Número básico de reprodução
        - 'cost': float - Valor final da função de custo
        - 'success': bool - Se a otimização foi bem-sucedida
        - 'message': str - Mensagem do otimizador
        - 'iterations': int - Número de iterações
        - 'metrics': dict - Métricas de qualidade do ajuste (R², RMSE)
        - 'prediction': dict - Predições do modelo ('S', 'E', 'I', 'R')
    """
    print("\n" + "="*60)
    print("OTIMIZAÇÃO DE PARÂMETROS SEIR")
    print("="*60)

    # Preparar chute inicial (3 parâmetros)
    x0 = [initial_guess['beta'], initial_guess['sigma'], initial_guess['gamma']]
    print(f"Chute inicial: beta={x0[0]:.4f}, sigma={x0[1]:.4f}, gamma={x0[2]:.4f}")

    # Preparar limites
    param_bounds = [bounds['beta'], bounds['sigma'], bounds['gamma']]
    print(f"Limites: beta={bounds['beta']}, sigma={bounds['sigma']}, gamma={bounds['gamma']}")

    # Callback para monitorar progresso
    iteration_count = [0]

    def callback(xk):
        iteration_count[0] += 1
        if iteration_count[0] % 10 == 0:
            cost = cost_function_seir(xk, time_data, E_data, I_data, R_data, N)
            print(f"Iteracao {iteration_count[0]}: beta={xk[0]:.6f}, sigma={xk[1]:.6f}, "
                  f"gamma={xk[2]:.6f}, Custo={cost:.2e}")

    print(f"\nIniciando otimização com método {method}...")

    # Otimização
    result = minimize(
        fun=cost_function_seir,
        x0=x0,
        args=(time_data, E_data, I_data, R_data, N),
        method=method,
        bounds=param_bounds,
        callback=callback,
        options={
            'maxiter': 1000,
            'ftol': 1e-9,
            'disp': False
        }
    )

    print(f"\nOtimizacao concluida: {result.message}")
    print(f"Total de iteracoes: {result.nit}")

    # Extrair parâmetros ótimos (3 parâmetros)
    beta_opt, sigma_opt, gamma_opt = result.x
    R0 = beta_opt / gamma_opt
    incubation_period = 1 / sigma_opt

    print(f"\nParametros estimados:")
    print(f"  beta = {beta_opt:.6f}")
    print(f"  sigma = {sigma_opt:.6f}")
    print(f"  gamma = {gamma_opt:.6f}")
    print(f"  R0 = {R0:.4f}")
    print(f"  Periodo de incubacao = {incubation_period:.1f} dias")

    # Simular com parâmetros ótimos para calcular métricas
    E0 = E_data[0]
    I0 = I_data[0]
    R0_data = R_data[0]
    S0 = N - E0 - I0 - R0_data

    from src.models.seir import SEIRModel
    model = SEIRModel(beta_opt, sigma_opt, gamma_opt, N)
    solution = model.simulate(
        initial_conditions=[S0, E0, I0, R0_data],
        t_span=(time_data[0], time_data[-1]),
        t_eval=time_data
    )

    E_pred = solution['E']
    I_pred = solution['I']
    R_pred = solution['R']
    S_pred = solution['S']

    # Calcular métricas de qualidade do ajuste

    # R² (coeficiente de determinação) - fit apenas I e R
    ss_res = np.sum((I_data - I_pred)**2) + np.sum((R_data - R_pred)**2)
    ss_tot = np.sum((I_data - np.mean(I_data))**2) + np.sum((R_data - np.mean(R_data))**2)

    if ss_tot > 0:
        r2 = 1 - (ss_res / ss_tot)
    else:
        r2 = 0.0

    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(ss_res / (2 * len(time_data)))

    # MAPE (Mean Absolute Percentage Error) para I
    mape_I = np.mean(np.abs((I_data - I_pred) / (I_data + 1))) * 100  # +1 para evitar divisão por zero

    print(f"\nMetricas de ajuste:")
    print(f"  R2 = {r2:.4f}")
    print(f"  RMSE = {rmse:.2f}")
    print(f"  MAPE (I) = {mape_I:.2f}%")

    return {
        'beta': beta_opt,
        'sigma': sigma_opt,
        'gamma': gamma_opt,
        'R0': R0,
        'cost': result.fun,
        'success': result.success,
        'message': result.message,
        'iterations': result.nit,
        'metrics': {
            'r2': r2,
            'rmse': rmse,
            'mape_I': mape_I,
            'rss': result.fun
        },
        'prediction': {
            'S': S_pred,
            'E': E_pred,
            'I': I_pred,
            'R': R_pred
        }
    }
