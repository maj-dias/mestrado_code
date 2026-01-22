"""
Identificação de Parâmetros do Modelo SIDARTHE

Este módulo implementa a identificação dos 12 parâmetros do modelo SIDARTHE
a partir de dados reais de COVID-19 usando otimização numérica.

Método: Otimização híbrida de duas fases
1. Fase Global: Differential Evolution (exploração)
2. Fase Local: L-BFGS-B (refinamento)

Dados necessários:
- Casos confirmados totais
- Mortes acumuladas
- Recuperados
- Casos ativos
- (Opcional) Hospitalizados

Função objetivo:
    J(θ) = Σᵢ wᵢ·MSE(yᵢ_obs, yᵢ_sim(θ))

Onde θ = [α, β, ε, ζ, η, λ, μ, ρ, θ, κ, ν, τ] (12 parâmetros)
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from typing import Dict, Tuple, Optional
import pandas as pd

from ..models.sidarthe import SIDARTHEModel


class SIDARTHEIdentifier:
    """
    Identifica os 12 parâmetros do modelo SIDARTHE a partir de dados reais

    Estratégia de otimização:
    1. Differential Evolution para busca global no espaço de parâmetros
    2. L-BFGS-B para refinamento local
    3. Minimização de erro quadrático médio ponderado entre dados e simulação

    Parâmetros
    ----------
    data : pd.DataFrame
        DataFrame com colunas: 'confirmed', 'deaths', 'recovered', 'active'
        Opcionalmente: 'hospitalized'
    N : float
        População total
    config : Dict
        Configuração com pesos, limites de parâmetros, etc.
    """

    def __init__(self, data: pd.DataFrame, N: float, config: Dict):
        self.data = data
        self.N = N
        self.config = config

        # Processar dados observados
        self.t_obs = np.arange(len(data))
        self.cases_obs = data['confirmed'].values.astype(float)
        self.deaths_obs = data['deaths'].values.astype(float)
        self.recovered_obs = data['recovered'].values.astype(float)
        self.active_obs = data['active'].values.astype(float)

        # Hospitalizados (se disponível)
        if 'hospitalized' in data.columns:
            self.hosp_obs = data['hospitalized'].values.astype(float)
        else:
            self.hosp_obs = None

        # Validar dados
        self._validate_data()

        # Contadores para monitoramento
        self.iteration_count = 0
        self.best_cost = np.inf

    def _validate_data(self):
        """Valida que os dados são consistentes"""
        if len(self.cases_obs) == 0:
            raise ValueError("Dados vazios")

        if np.any(self.cases_obs < 0):
            raise ValueError("Casos confirmados não podem ser negativos")

        if np.any(self.deaths_obs < 0):
            raise ValueError("Mortes não podem ser negativas")

        # Mortes não podem exceder casos confirmados
        if np.any(self.deaths_obs > self.cases_obs):
            raise ValueError("Mortes não podem exceder casos confirmados")

    def objective_function(self, params_array: np.ndarray) -> float:
        """
        Função objetivo: soma ponderada de erros quadráticos médios

        Minimizar:
            J = w1·MSE(casos) + w2·MSE(mortes) +
                w3·MSE(curados) + w4·MSE(ativos) +
                w5·MSE(hospitalizados)  [se disponível]

        Parâmetros
        ----------
        params_array : np.ndarray
            Array com 12 parâmetros [α, β, ε, ζ, η, λ, μ, ρ, θ, κ, ν, τ]

        Retorna
        -------
        float
            Custo total (menor é melhor)
        """
        try:
            # Converter array para dicionário
            params = self._array_to_params(params_array)

            # Estimar condições iniciais
            y0 = self._estimate_initial_conditions(params)

            # Criar e simular modelo
            model = SIDARTHEModel(params, self.N)
            sol = model.simulate(
                y0=y0,
                t_span=(0, len(self.data)),
                t_eval=self.t_obs
            )

            # Calcular observáveis simulados
            obs_sim = model.compute_observables(sol)

            # Calcular erros quadráticos médios
            mse_cases = np.mean((obs_sim['confirmed'] - self.cases_obs)**2)
            mse_deaths = np.mean((obs_sim['deaths'] - self.deaths_obs)**2)
            mse_recovered = np.mean((obs_sim['recovered'] - self.recovered_obs)**2)
            mse_active = np.mean((obs_sim['active'] - self.active_obs)**2)

            # Pesos configuráveis
            weights = self.config['optimization']['weights']
            cost = (weights['cases'] * mse_cases +
                   weights['deaths'] * mse_deaths +
                   weights['recovered'] * mse_recovered +
                   weights['active'] * mse_active)

            # Adicionar hospitalização se disponível
            if self.hosp_obs is not None:
                mse_hosp = np.mean((obs_sim['hospitalized'] - self.hosp_obs)**2)
                cost += weights.get('hospitalized', 1.0) * mse_hosp

            # Monitorar progresso
            self.iteration_count += 1
            if cost < self.best_cost:
                self.best_cost = cost
                if self.iteration_count % 10 == 0:
                    print(f"  Iteração {self.iteration_count:4d}: Custo = {cost:.4e}")

            return cost

        except Exception as e:
            # Penalizar parâmetros que causam erro
            # (ex: populações negativas, overflow, etc.)
            return 1e15

    def identify(self) -> Dict:
        """
        Executa identificação de parâmetros usando otimização híbrida

        Fase 1: Differential Evolution (global)
        Fase 2: L-BFGS-B (local)

        Retorna
        -------
        dict
            Dicionário com:
            - 'parameters': Parâmetros identificados
            - 'cost': Custo final
            - 'metrics': Métricas de ajuste (R², MAPE, etc.)
            - 'optimization': Informações da otimização
        """
        print("\n" + "="*70)
        print("IDENTIFICAÇÃO DE PARÂMETROS - MODELO SIDARTHE")
        print("="*70)

        # Limites dos parâmetros
        bounds = self._get_parameter_bounds()

        # FASE 1: Busca Global (Differential Evolution)
        print("\nFASE 1: Busca Global (Differential Evolution)")
        print("-" * 70)
        print(f"Iterações máximas: {self.config['optimization']['global_maxiter']}")
        print(f"População: 15 indivíduos")
        print(f"Parâmetros: 12 (alpha, beta, epsilon, zeta, eta, lambda, mu, rho, theta, kappa, nu, tau)")

        self.iteration_count = 0
        self.best_cost = np.inf

        result_global = differential_evolution(
            func=self.objective_function,
            bounds=bounds,
            maxiter=self.config['optimization']['global_maxiter'],
            popsize=15,
            seed=42,
            disp=False,  # Usar nosso próprio monitoramento
            polish=False,  # Não fazer refinamento automático
            workers=1  # Não paralelizar (para manter print ordenado)
        )

        print(f"\nFase 1 concluída!")
        print(f"  Custo global: {result_global.fun:.4e}")
        print(f"  Iterações: {self.iteration_count}")

        # FASE 2: Refinamento Local (L-BFGS-B)
        print("\nFASE 2: Refinamento Local (L-BFGS-B)")
        print("-" * 70)

        self.iteration_count = 0
        self.best_cost = result_global.fun

        result_local = minimize(
            fun=self.objective_function,
            x0=result_global.x,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.config['optimization']['local_maxiter']}
        )

        print(f"\nFase 2 concluída!")
        print(f"  Custo local: {result_local.fun:.4e}")
        print(f"  Iterações: {self.iteration_count}")
        print(f"  Sucesso: {result_local.success}")

        # Parâmetros finais
        params_final = self._array_to_params(result_local.x)

        print("\n" + "="*70)
        print("PARÂMETROS IDENTIFICADOS")
        print("="*70)
        for name, value in params_final.items():
            print(f"  {name:12s} = {value:.6f}")

        # Computar métricas de ajuste
        metrics = self._compute_metrics(params_final)

        print(f"\nMÉTRICAS DE AJUSTE:")
        print(f"  R² (casos):      {metrics['R2_cases']:.4f}")
        print(f"  R² (mortes):     {metrics['R2_deaths']:.4f}")
        print(f"  MAPE (casos):    {metrics['MAPE_cases']:.2f}%")
        print(f"  MAPE (mortes):   {metrics['MAPE_deaths']:.2f}%")
        print(f"  R₀ estimado:     {metrics['R0']:.4f}")

        return {
            'parameters': params_final,
            'cost': result_local.fun,
            'metrics': metrics,
            'optimization': {
                'global_cost': result_global.fun,
                'local_cost': result_local.fun,
                'success': result_local.success,
                'global_iterations': result_global.nit,
                'local_iterations': result_local.nit
            }
        }

    def _array_to_params(self, arr: np.ndarray) -> Dict[str, float]:
        """
        Converte array numpy para dicionário de parâmetros

        Parâmetros
        ----------
        arr : np.ndarray
            Array com 12 elementos

        Retorna
        -------
        dict
            Dicionário com nomes dos parâmetros
        """
        param_names = ['alpha', 'beta', 'epsilon', 'zeta', 'eta',
                       'lambda_', 'mu', 'rho', 'theta', 'kappa', 'nu', 'tau']

        if len(arr) != 12:
            raise ValueError(f"Array deve ter 12 elementos, recebido: {len(arr)}")

        return {name: float(arr[i]) for i, name in enumerate(param_names)}

    def _get_parameter_bounds(self) -> list:
        """
        Define limites dos 12 parâmetros

        Baseado em:
        - Plausibilidade biológica
        - Literatura (Giordano et al. 2020)
        - Restrições físicas

        Retorna
        -------
        list
            Lista de 12 tuplas (min, max)
        """
        return [
            (0.01, 1.0),   # alpha: transmissão assintomáticos
            (0.01, 1.0),   # beta: transmissão sintomáticos
            (0.001, 0.5),  # epsilon: detecção assintomáticos
            (0.05, 0.5),   # zeta: desenvolvimento sintomas (~2-20 dias)
            (0.001, 0.5),  # eta: detecção sintomáticos
            (0.01, 0.3),   # lambda: cura não detectados (~3-100 dias)
            (0.01, 0.3),   # mu: cura sintomáticos não detectados
            (0.01, 0.3),   # rho: cura detectados
            (0.001, 0.1),  # theta: agravamento D→T (~1-10%)
            (0.001, 0.1),  # kappa: agravamento R→T
            (0.01, 0.3),   # nu: cura casos graves (~3-100 dias)
            (0.001, 0.05)  # tau: mortalidade (~0.1-5% por dia)
        ]

    def _estimate_initial_conditions(self, params: Dict[str, float]) -> np.ndarray:
        """
        Estima condições iniciais [S0, I0, D0, A0, R0, T0, H0, E0]

        Problema: I, D, A, R, T não são diretamente observados no dia 0!

        Solução: Heurística baseada em dados observados

        Parâmetros
        ----------
        params : dict
            Parâmetros do modelo (podem influenciar estimativa)

        Retorna
        -------
        np.ndarray
            Array com 8 condições iniciais
        """
        # Observados no dia 0
        cases0 = self.cases_obs[0]
        deaths0 = self.deaths_obs[0]
        recovered0 = self.recovered_obs[0]
        active0 = self.active_obs[0]

        # E0: Mortes no dia 0
        E0 = deaths0

        # H0: Recuperados no dia 0
        H0 = recovered0

        # T0: Estimativa de graves (~5% dos casos confirmados ativos)
        if self.hosp_obs is not None:
            T0 = self.hosp_obs[0]
        else:
            T0 = 0.05 * active0

        # Casos detectados (D + R + T) ≈ active0
        # Distribuir entre assintomáticos detectados (D) e sintomáticos detectados (R)
        # Heurística: 70% assintomáticos, 30% sintomáticos entre os ativos não graves
        non_threatened_active = max(0, active0 - T0)
        D0 = 0.7 * non_threatened_active
        R0_comp = 0.3 * non_threatened_active

        # I0 e A0: Casos não detectados (estimativa)
        # Assumir que há tantos não detectados quanto detectados (fator de subnotificação ~2)
        I0 = D0  # Assintomáticos não detectados ≈ assintomáticos detectados
        A0 = R0_comp  # Sintomáticos não detectados ≈ sintomáticos detectados

        # S0: População restante
        S0 = self.N - (I0 + D0 + A0 + R0_comp + T0 + H0 + E0)

        # Garantir não-negatividade
        S0 = max(0, S0)
        I0 = max(1, I0)  # Pelo menos 1 infectado
        D0 = max(0, D0)
        A0 = max(1, A0)  # Pelo menos 1 sintomático
        R0_comp = max(0, R0_comp)
        T0 = max(0, T0)

        return np.array([S0, I0, D0, A0, R0_comp, T0, H0, E0])

    def _compute_metrics(self, params: Dict[str, float]) -> Dict:
        """
        Calcula métricas de ajuste

        Métricas:
        - R² (coeficiente de determinação)
        - MAPE (erro percentual absoluto médio)
        - R₀ (número reprodutivo básico)

        Parâmetros
        ----------
        params : dict
            Parâmetros identificados

        Retorna
        -------
        dict
            Dicionário com métricas
        """
        # Simular com parâmetros identificados
        y0 = self._estimate_initial_conditions(params)
        model = SIDARTHEModel(params, self.N)
        sol = model.simulate(y0, (0, len(self.data)), self.t_obs)
        obs_sim = model.compute_observables(sol)

        # R² (coeficiente de determinação)
        def compute_r2(y_obs, y_sim):
            ss_res = np.sum((y_obs - y_sim)**2)
            ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
            return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # MAPE (erro percentual absoluto médio)
        def compute_mape(y_obs, y_sim):
            # Evitar divisão por zero
            mask = y_obs > 0
            if np.sum(mask) == 0:
                return np.inf
            return 100 * np.mean(np.abs((y_obs[mask] - y_sim[mask]) / y_obs[mask]))

        metrics = {
            'R2_cases': compute_r2(self.cases_obs, obs_sim['confirmed']),
            'R2_deaths': compute_r2(self.deaths_obs, obs_sim['deaths']),
            'R2_recovered': compute_r2(self.recovered_obs, obs_sim['recovered']),
            'R2_active': compute_r2(self.active_obs, obs_sim['active']),
            'MAPE_cases': compute_mape(self.cases_obs, obs_sim['confirmed']),
            'MAPE_deaths': compute_mape(self.deaths_obs, obs_sim['deaths']),
            'MAPE_recovered': compute_mape(self.recovered_obs, obs_sim['recovered']),
            'MAPE_active': compute_mape(self.active_obs, obs_sim['active']),
            'R0': model.compute_R0()
        }

        if self.hosp_obs is not None:
            metrics['R2_hospitalized'] = compute_r2(self.hosp_obs, obs_sim['hospitalized'])
            metrics['MAPE_hospitalized'] = compute_mape(self.hosp_obs, obs_sim['hospitalized'])

        return metrics
