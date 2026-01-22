# Plano de Implementação: Controle Ótimo para Modelo SEIR

## 1. Introdução

Este documento descreve o plano para introduzir **controle ótimo** no modelo SEIR de COVID-19, posicionando o trabalho claramente dentro da área de **Engenharia Elétrica - Controle e Automação**.

### Objetivo

Projetar políticas ótimas de intervenção (distanciamento social e vacinação) que minimizem o impacto da epidemia enquanto consideram os custos econômicos e logísticos das intervenções.

---

## 2. Formulação Matemática

### 2.1 Sistema SEIR com Controle

O modelo SEIR original (sem controle) é dado por:

```
dS/dt = -β·S·I/N
dE/dt = β·S·I/N - σ·E
dI/dt = σ·E - γ·I
dR/dt = γ·I
```

**Introduzindo variáveis de controle u(t) = [u₁(t), u₂(t)]:**

```
dS/dt = -β·(1-u₁)·S·I/N - u₂·S
dE/dt = β·(1-u₁)·S·I/N - σ·E
dI/dt = σ·E - γ·I
dR/dt = γ·I + u₂·S
```

**Variáveis de controle:**

- **u₁(t) ∈ [0, u₁ₘₐₓ]**: Intensidade de distanciamento social/lockdown
  - u₁ = 0: Sem restrições (transmissão normal β)
  - u₁ = 1: Lockdown total (transmissão zero)
  - Limites práticos: u₁ₘₐₓ ∈ [0.7, 0.9] (lockdown total é inviável)

- **u₂(t) ∈ [0, u₂ₘₐₓ]**: Taxa de vacinação (pessoas/dia)
  - u₂ = 0: Sem vacinação
  - u₂ₘₐₓ: Capacidade máxima de vacinação (ex: 5000 pessoas/dia)
  - Limites práticos: u₂ₘₐₓ depende da infraestrutura disponível

**Interpretação física:**
- β·(1-u₁): Lockdown reduz a taxa de transmissão efetiva
- u₂·S: Vacinação move suscetíveis diretamente para removidos

### 2.2 Problema de Controle Ótimo

**Objetivo:** Minimizar o funcional de custo J ao longo de um horizonte T (dias):

```
J(u) = ∫₀ᵀ L(x(t), u(t), t) dt + Φ(x(T))

onde:

L(x, u, t) = w₁·I(t) + w₂·u₁²(t) + w₃·u₂²(t)   (custo instantâneo)
Φ(x(T)) = wf·I(T)                                (custo terminal)
```

**Componentes do custo:**

1. **w₁·I(t)**: Custo de infecções ativas
   - Representa impacto na saúde pública, óbitos, hospitalizações
   - Peso típico: w₁ = 1e6 (alto, prioriza reduzir infecções)

2. **w₂·u₁²(t)**: Custo de lockdown
   - Representa impacto econômico (desemprego, PIB)
   - Quadrático para penalizar mudanças bruscas
   - Peso típico: w₂ = 1e3

3. **w₃·u₂²(t)**: Custo de vacinação
   - Representa custo logístico e financeiro
   - Quadrático para evitar picos de demanda
   - Peso típico: w₃ = 1e2

4. **wf·I(T)**: Penalidade terminal
   - Incentiva reduzir infecções ao final do horizonte
   - Evita que o controle "empurre" infecções para depois de T

**Trade-offs:**
- Aumentar w₁ → mais lockdown/vacinação, menos infecções
- Aumentar w₂ → menos lockdown, mais infecções
- Aumentar w₃ → menos vacinação, mais infecções

---

## 3. Métodos de Solução

### 3.1 Princípio do Máximo de Pontryagin (PMP)

**Abordagem clássica de controle ótimo baseada em cálculo variacional.**

#### Hamiltoniano

```
H(x, u, λ, t) = L(x, u, t) + λᵀ·f(x, u, t)

H = w₁·I + w₂·u₁² + w₃·u₂²
    + λS·(-β(1-u₁)SI/N - u₂S)
    + λE·(β(1-u₁)SI/N - σE)
    + λI·(σE - γI)
    + λR·(γI + u₂S)
```

onde:
- **x = [S, E, I, R]**: Variáveis de estado
- **u = [u₁, u₂]**: Variáveis de controle
- **λ = [λS, λE, λI, λR]**: Variáveis adjuntas (co-estados)

#### Condições de Otimalidade

**1. Sistema de Estados (forward, t: 0 → T):**
```
dx/dt = ∂H/∂λ = f(x, u*, t)
x(0) = x₀  (condições iniciais conhecidas)
```

**2. Sistema de Co-estados (backward, t: T → 0):**
```
dλS/dt = -∂H/∂S = β(1-u₁)I/N·(λE - λS) + u₂·(λR - λS)
dλE/dt = -∂H/∂E = σ·(λI - λE)
dλI/dt = -∂H/∂I = -w₁ + β(1-u₁)S/N·(λE - λS) + γ·(λR - λI)
dλR/dt = -∂H/∂R = 0

λ(T) = ∂Φ/∂x|ₜ₌ᵀ = [0, 0, wf, 0]ᵀ  (condições terminais)
```

**3. Condição de Estacionariedade (controle ótimo):**
```
∂H/∂u₁ = 0  →  2w₂·u₁ + βSI/N·(λS - λE) = 0
              →  u₁* = -βSI/(2w₂N)·(λS - λE)

∂H/∂u₂ = 0  →  2w₃·u₂ + S·(λR - λS) = 0
              →  u₂* = -S/(2w₃)·(λR - λS)
```

Com projeção nos limites:
```
u₁*(t) = min(u₁ₘₐₓ, max(0, -βSI/(2w₂N)·(λS - λE)))
u₂*(t) = min(u₂ₘₐₓ, max(0, -S/(2w₃)·(λR - λS)))
```

#### Problema de Valor de Contorno de Duas Pontas (TPBVP)

- **x(0)** é conhecido (condições iniciais dos dados)
- **λ(T)** é conhecido (condições terminais do custo)
- Deve-se resolver simultaneamente:
  - Equações de estado (forward)
  - Equações adjuntas (backward)
  - Condições de controle ótimo

**Métodos numéricos:**

1. **Shooting Method** (método de tiro):
   - Chutar λ(0)
   - Integrar estados e co-estados forward
   - Verificar se λ(T) satisfaz condições terminais
   - Iterar até convergência (Newton-Raphson)

2. **Collocation Method** (colocação):
   - Discretizar t em malha [t₀, t₁, ..., tₙ]
   - Transformar ODE em sistema algébrico não-linear
   - Resolver todas as equações simultaneamente

### 3.2 Programação Dinâmica (Bellman)

**Abordagem alternativa baseada na equação de Hamilton-Jacobi-Bellman (HJB).**

#### Equação de Bellman

```
V(x, t) = min_u [L(x, u, t)·Δt + V(f(x, u, t)·Δt + x, t+Δt)]

onde V(x, t) é a função valor (custo ótimo de t até T)
```

**Condição de otimalidade:**
```
u*(x, t) = argmin_u [L(x, u, t) + (∂V/∂x)ᵀ·f(x, u, t)]
```

**Implementação numérica:**
- Discretizar estado x em malha 4D (S, E, I, R)
- Discretizar tempo t em passos Δt
- Resolver backward de t=T até t=0 usando programação dinâmica

**Desvantagens:**
- Maldição da dimensionalidade (4 estados + tempo)
- Requer interpolação multidimensional
- Mais caro computacionalmente que PMP para este problema

---

## 4. Arquitetura de Implementação

### 4.1 Estrutura de Diretórios

```
src/
├── models/
│   ├── seir.py                    # Modelo SEIR sem controle (já existe)
│   └── seir_controlled.py         # NOVO: Modelo SEIR com controles u₁, u₂
├── control/
│   ├── __init__.py
│   ├── optimal_control.py         # NOVO: Classe base OptimalControl
│   ├── pontryagin.py              # NOVO: Solver PMP (shooting/collocation)
│   ├── cost_functions.py          # NOVO: Definições de L(x,u,t) e Φ(x)
│   └── constraints.py             # NOVO: Limites e restrições de controle
├── utils/
│   ├── data_loader.py             # Já existe
│   └── bvp_solvers.py             # NOVO: Solvers para TPBVP
└── plots/
    ├── identification.py          # Já existe
    └── control_plots.py           # NOVO: Visualização de trajetórias ótimas

config/
├── seir.yaml                      # Já existe (identificação)
└── optimal_control.yaml           # NOVO: Configuração de controle ótimo

scripts/
├── identification_seir.py         # Já existe
└── optimal_control_seir.py        # NOVO: Script principal de otimização

results/
├── parameters/
│   └── seir_params.json           # Já existe (β, σ, γ identificados)
├── control/
│   ├── optimal_trajectories.json  # NOVO: u₁*(t), u₂*(t)
│   └── controlled_states.json     # NOVO: S(t), E(t), I(t), R(t) sob controle
└── figures/
    └── control/                   # NOVO: Gráficos de controle ótimo
```

### 4.2 Arquivos Novos

#### **src/models/seir_controlled.py**

```python
class SEIRControlledModel:
    """
    Modelo SEIR com variáveis de controle u₁ (lockdown) e u₂ (vacinação)
    """
    def __init__(self, beta: float, sigma: float, gamma: float, N: float):
        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.N = N

    def derivatives(self, t: float, y: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calcula dX/dt com controles u = [u₁, u₂]

        Parâmetros
        ----------
        y : [S, E, I, R]
        u : [u₁, u₂] - controles no instante t

        Retorna
        -------
        [dS, dE, dI, dR]
        """
        S, E, I, R = y
        u1, u2 = u

        infection_rate = self.beta * (1 - u1) * S * I / self.N

        dS = -infection_rate - u2 * S
        dE = infection_rate - self.sigma * E
        dI = self.sigma * E - self.gamma * I
        dR = self.gamma * I + u2 * S

        return np.array([dS, dE, dI, dR])

    def simulate_with_control(
        self,
        initial_conditions: np.ndarray,
        control_trajectory: Callable,  # u(t) -> [u1, u2]
        t_span: Tuple[float, float],
        t_eval: np.ndarray
    ) -> Dict:
        """Simula sistema com trajetória de controle dada"""
        # Implementação usando solve_ivp com controle u(t)
        pass
```

#### **src/control/cost_functions.py**

```python
class QuadraticCost:
    """
    Função de custo quadrática:
    L(x, u, t) = w₁·I + w₂·u₁² + w₃·u₂²
    Φ(x) = wf·I(T)
    """
    def __init__(self, w1: float, w2: float, w3: float, wf: float):
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.wf = wf

    def running_cost(self, x: np.ndarray, u: np.ndarray, t: float) -> float:
        """Custo instantâneo L(x, u, t)"""
        S, E, I, R = x
        u1, u2 = u
        return self.w1 * I + self.w2 * u1**2 + self.w3 * u2**2

    def terminal_cost(self, x: np.ndarray) -> float:
        """Custo terminal Φ(x)"""
        S, E, I, R = x
        return self.wf * I

    def terminal_costates(self, x: np.ndarray) -> np.ndarray:
        """Condição terminal λ(T) = ∂Φ/∂x"""
        return np.array([0, 0, self.wf, 0])
```

#### **src/control/pontryagin.py**

```python
class PontryaginSolver:
    """
    Resolve problema de controle ótimo usando Princípio do Máximo de Pontryagin
    """
    def __init__(
        self,
        model: SEIRControlledModel,
        cost: QuadraticCost,
        u_bounds: Dict[str, Tuple[float, float]]
    ):
        self.model = model
        self.cost = cost
        self.u_bounds = u_bounds

    def hamiltonian(self, x, u, lam, t):
        """Calcula Hamiltoniano H(x, u, λ, t)"""
        L = self.cost.running_cost(x, u, t)
        f = self.model.derivatives(t, x, u)
        return L + np.dot(lam, f)

    def costate_derivatives(self, x, u, lam, t):
        """
        Calcula dλ/dt = -∂H/∂x

        Retorna
        -------
        [dλS, dλE, dλI, dλR]
        """
        # Derivadas analíticas do Hamiltoniano
        pass

    def optimal_control(self, x, lam, t):
        """
        Calcula u*(t) = argmin_u H(x, u, λ, t)

        Usando condições de primeira ordem:
        u₁* = min(max(0, -βSI/(2w₂N)·(λS - λE)), u₁ₘₐₓ)
        u₂* = min(max(0, -S/(2w₃)·(λR - λS)), u₂ₘₐₓ)
        """
        S, E, I, R = x
        lam_S, lam_E, lam_I, lam_R = lam

        # Controle sem limites
        u1_unbounded = -self.model.beta * S * I / (2 * self.cost.w2 * self.model.N) * (lam_S - lam_E)
        u2_unbounded = -S / (2 * self.cost.w3) * (lam_R - lam_S)

        # Projetar em limites
        u1 = np.clip(u1_unbounded, self.u_bounds['u1'][0], self.u_bounds['u1'][1])
        u2 = np.clip(u2_unbounded, self.u_bounds['u2'][0], self.u_bounds['u2'][1])

        return np.array([u1, u2])

    def solve_shooting(self, x0, T, n_points=100):
        """
        Resolve TPBVP usando shooting method

        Algoritmo:
        1. Chutar λ(0)
        2. Integrar forward: [x(t), λ(t)] de 0 a T
        3. Verificar se λ(T) = ∂Φ/∂x|T
        4. Ajustar λ(0) usando Newton-Raphson
        5. Repetir até convergência
        """
        pass

    def solve(self, x0, T, method='shooting', n_points=100):
        """Interface principal de solução"""
        if method == 'shooting':
            return self.solve_shooting(x0, T, n_points)
        elif method == 'collocation':
            return self.solve_collocation(x0, T, n_points)
        else:
            raise ValueError(f"Método desconhecido: {method}")
```

#### **config/optimal_control.yaml**

```yaml
# Configuração de Controle Ótimo - SEIR COVID-19

# Parâmetros do modelo (carregar de identificação)
model:
  parameters_file: "results/parameters/seir_params.json"

# Variáveis de controle
control:
  u1_lockdown:
    description: "Intensidade de distanciamento social"
    min: 0.0
    max: 0.8       # Lockdown total (1.0) é inviável

  u2_vaccination:
    description: "Taxa de vacinação (pessoas/dia)"
    min: 0.0
    max: 5000.0    # Capacidade máxima de vacinação

# Função de custo
cost:
  weights:
    w1_infections: 1.0e6     # Peso de infecções I(t)
    w2_lockdown: 1.0e3       # Peso de lockdown u₁²
    w3_vaccination: 1.0e2    # Peso de vacinação u₂²
    wf_terminal: 5.0e6       # Peso terminal I(T)

  horizon_days: 365          # Horizonte de controle (1 ano)

# Solver PMP
solver:
  method: "shooting"         # Opções: shooting, collocation
  n_time_points: 365         # Discretização temporal
  tolerance: 1.0e-6
  max_iterations: 100

# Condições iniciais (do final da identificação)
initial_conditions:
  use_last_data_point: true  # Usar último ponto dos dados

# Saída
output:
  results_dir: "results/control"
  save_trajectories: true
  save_figures: true

  figures:
    format: "png"
    dpi: 300
```

#### **scripts/optimal_control_seir.py**

```python
#!/usr/bin/env python3
"""
Controle Ótimo do Modelo SEIR - COVID-19 Brasil

Este script resolve o problema de controle ótimo:
1. Carregar parâmetros identificados (β, σ, γ)
2. Definir problema de controle (custos, limites)
3. Resolver usando Pontryagin Maximum Principle
4. Gerar trajetórias ótimas de controle u₁*(t), u₂*(t)
5. Simular sistema controlado S(t), E(t), I(t), R(t)
6. Visualizar e salvar resultados

Uso:
    python scripts/optimal_control_seir.py
"""

import yaml
import json
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.seir_controlled import SEIRControlledModel
from src.control.pontryagin import PontryaginSolver
from src.control.cost_functions import QuadraticCost
from src.plots.control_plots import plot_optimal_control

def main():
    # 1. Carregar configuração
    config = load_config('config/optimal_control.yaml')

    # 2. Carregar parâmetros identificados
    params = load_identified_parameters('results/parameters/seir_params.json')
    beta, sigma, gamma = params['beta'], params['sigma'], params['gamma']
    N = params['config']['data']['population']

    print("="*70)
    print("CONTROLE ÓTIMO SEIR - COVID-19 BRASIL")
    print("="*70)
    print(f"Parâmetros identificados:")
    print(f"  β = {beta:.6f}")
    print(f"  σ = {sigma:.6f}")
    print(f"  γ = {gamma:.6f}")
    print(f"  R₀ = {beta/gamma:.4f}")

    # 3. Criar modelo controlado
    model = SEIRControlledModel(beta, sigma, gamma, N)

    # 4. Definir função de custo
    cost = QuadraticCost(
        w1=config['cost']['weights']['w1_infections'],
        w2=config['cost']['weights']['w2_lockdown'],
        w3=config['cost']['weights']['w3_vaccination'],
        wf=config['cost']['weights']['wf_terminal']
    )

    # 5. Condições iniciais (último ponto dos dados)
    x0 = get_initial_conditions(params)
    T = config['cost']['horizon_days']

    print(f"\nCondições iniciais (t=0):")
    print(f"  S₀ = {x0[0]:,.0f}")
    print(f"  E₀ = {x0[1]:,.0f}")
    print(f"  I₀ = {x0[2]:,.0f}")
    print(f"  R₀ = {x0[3]:,.0f}")
    print(f"\nHorizonte: T = {T} dias")

    # 6. Resolver controle ótimo
    print(f"\nResolvendo problema de controle ótimo...")
    print(f"Método: {config['solver']['method']}")

    solver = PontryaginSolver(
        model=model,
        cost=cost,
        u_bounds={
            'u1': (config['control']['u1_lockdown']['min'],
                   config['control']['u1_lockdown']['max']),
            'u2': (config['control']['u2_vaccination']['min'],
                   config['control']['u2_vaccination']['max'])
        }
    )

    solution = solver.solve(
        x0=x0,
        T=T,
        method=config['solver']['method'],
        n_points=config['solver']['n_time_points']
    )

    # 7. Exibir resultados
    print("\n" + "="*70)
    print("RESULTADOS DO CONTROLE ÓTIMO")
    print("="*70)

    total_cost = solution['total_cost']
    print(f"Custo total J: {total_cost:.2e}")

    # Métricas de controle
    u1_mean = np.mean(solution['u1'])
    u2_mean = np.mean(solution['u2'])
    I_max = np.max(solution['I'])
    I_final = solution['I'][-1]

    print(f"\nMétricas de controle:")
    print(f"  Lockdown médio: {u1_mean*100:.1f}%")
    print(f"  Vacinação média: {u2_mean:,.0f} pessoas/dia")
    print(f"  Pico de infecções: {I_max:,.0f}")
    print(f"  Infecções finais I(T): {I_final:,.0f}")

    # 8. Salvar resultados
    save_results(solution, config)

    # 9. Gerar gráficos
    plot_optimal_control(solution, config)

    print("\n✅ CONTROLE ÓTIMO CONCLUÍDO!")
    return 0

if __name__ == '__main__':
    sys.exit(main())
```

---

## 5. Workflow de Execução

### 5.1 Sequência de Passos

```bash
# Passo 1: Identificar parâmetros (já feito)
python scripts/identification_seir.py
# Output: results/parameters/seir_params.json

# Passo 2: Resolver controle ótimo
python scripts/optimal_control_seir.py
# Output:
#   - results/control/optimal_trajectories.json
#   - results/control/controlled_states.json
#   - results/figures/control/*.png

# Passo 3: Análise de sensibilidade (opcional)
python scripts/sensitivity_analysis.py
# Varia w₁, w₂, w₃ e analisa impacto nas trajetórias
```

### 5.2 Resultados Esperados

**Arquivo: results/control/optimal_trajectories.json**
```json
{
  "time": [0, 1, 2, ..., 365],
  "u1_lockdown": [0.35, 0.42, 0.51, ..., 0.12],
  "u2_vaccination": [0, 0, 100, 500, ..., 3000],
  "total_cost": 1.234e9,
  "cost_breakdown": {
    "infections": 8.5e8,
    "lockdown": 3.2e8,
    "vaccination": 5.4e7
  }
}
```

**Arquivo: results/control/controlled_states.json**
```json
{
  "time": [0, 1, 2, ..., 365],
  "S": [...],
  "E": [...],
  "I": [...],
  "R": [...],
  "comparison_no_control": {
    "I": [...],  // Infecções sem controle
    "peak_reduction": 0.45  // 45% redução no pico
  }
}
```

---

## 6. Visualizações

### 6.1 Gráficos Principais

**Figura 1: Trajetórias Ótimas de Controle**
- **Subplot 1**: u₁*(t) - Intensidade de lockdown ao longo do tempo
- **Subplot 2**: u₂*(t) - Taxa de vacinação ao longo do tempo

**Figura 2: Estados Controlados vs Não-Controlados**
- **Subplot 1**: I(t) com controle vs sem controle
- **Subplot 2**: R(t) comparação
- Mostrar redução no pico de infecções

**Figura 3: Evolução dos 4 Compartimentos**
- S(t), E(t), I(t), R(t) sob controle ótimo

**Figura 4: Variáveis Adjuntas (Co-estados)**
- λS(t), λE(t), λI(t), λR(t)
- Interpretação: "preços sombra" dos compartimentos

**Figura 5: Hamiltoniano ao Longo do Tempo**
- H(t) deve ser constante ao longo da trajetória ótima (verificação numérica)

---

## 7. Extensões Futuras

### 7.1 Model Predictive Control (MPC)

Implementar controle em **horizonte deslizante**:
- A cada Δt dias, re-otimizar com novos dados
- Horizonte fixo de T dias à frente
- Mais robusto a incertezas e mudanças no sistema

### 7.2 Controle Robusto

Considerar **incertezas nos parâmetros**:
- β ∈ [β_min, β_max] (incerteza na taxa de transmissão)
- Minimizar pior caso (worst-case optimization)
- Usar teoria de jogos diferenciais

### 7.3 Controle Estocástico

Modelar **ruído estocástico**:
- dI = (σE - γI)dt + σ_I·dW  (Wiener process)
- Usar controle estocástico ótimo (HJB estocástico)

### 7.4 Multi-objetivo

Usar **otimização de Pareto**:
- Objetivo 1: Minimizar infecções
- Objetivo 2: Minimizar custo econômico
- Objetivo 3: Minimizar desigualdade social
- Gerar fronteira de Pareto de soluções

---

## 8. Conexão com Engenharia Elétrica

### Por que isso é Engenharia de Controle?

1. **Teoria de Controle Ótimo**
   - PMP é conteúdo central de disciplinas de pós-graduação em EE
   - Disciplinas: Controle Ótimo, Controle Avançado, Teoria de Sistemas

2. **Problema de Otimização Dinâmica**
   - Minimizar funcional de custo sujeito a dinâmica
   - Análogo a controle de processos industriais

3. **Two-Point Boundary Value Problem**
   - Técnica numérica clássica em controle ótimo
   - Shooting method, collocation

4. **Variáveis de Estado e Co-Estado**
   - λ representa "preços sombra" (interpretação econômica)
   - Dualidade estado-co-estado é fundamento de teoria de controle

5. **Trade-offs e Restrições**
   - Balancear múltiplos objetivos conflitantes
   - Respeitar limites físicos (u_min, u_max)
   - Típico de problemas de controle de processos

### Comparação com Outras Áreas

| Aspecto | Epidemiologia Clássica | Engenharia de Controle |
|---------|------------------------|------------------------|
| Foco | Previsão | Intervenção ótima |
| Metodologia | Ajuste de curvas | Otimização dinâmica |
| Ferramentas | Regressão, ML | Cálculo variacional, PMP |
| Output | Parâmetros β, γ | Políticas u*(t) |
| Objetivo | Entender | Controlar |

---

## 9. Cronograma de Implementação

### Fase 1: Infraestrutura Básica (1 semana)

- [ ] Implementar `SEIRControlledModel` (1 dia)
- [ ] Implementar `QuadraticCost` (0.5 dia)
- [ ] Criar `config/optimal_control.yaml` (0.5 dia)
- [ ] Implementar funções auxiliares (BVP solvers básicos) (2 dias)
- [ ] Testar modelo controlado com controles constantes (1 dia)

### Fase 2: Solver PMP (2 semanas)

- [ ] Derivar analiticamente ∂H/∂x (co-estados) (1 dia)
- [ ] Implementar `PontryaginSolver.costate_derivatives()` (2 dias)
- [ ] Implementar `PontryaginSolver.optimal_control()` (1 dia)
- [ ] Implementar shooting method básico (3 dias)
- [ ] Testes e debugging (3 dias)
- [ ] Validação com casos simples (2 dias)

### Fase 3: Aplicação COVID-19 (1 semana)

- [ ] Implementar `scripts/optimal_control_seir.py` (2 dias)
- [ ] Rodar com dados reais do Brasil (1 dia)
- [ ] Análise de sensibilidade de pesos w₁, w₂, w₃ (2 dias)
- [ ] Comparação com cenário sem controle (1 dia)
- [ ] Documentação de resultados (1 dia)

### Fase 4: Visualização e Relatório (1 semana)

- [ ] Implementar `src/plots/control_plots.py` (3 dias)
- [ ] Gerar todas as figuras (1 dia)
- [ ] Escrever seção de controle ótimo na dissertação (2 dias)
- [ ] Revisão e formatação (1 dia)

**Total estimado: 5 semanas**

---

## 10. Referências

### Livros

1. **Kirk, D. E.** (2004). *Optimal Control Theory: An Introduction*. Dover.
   - Capítulo 4: Princípio do Máximo de Pontryagin

2. **Lewis, F. L., Vrabie, D., Syrmos, V. L.** (2012). *Optimal Control*. Wiley.
   - Capítulo 2: Cálculo Variacional e Condições de Otimalidade

3. **Liberzon, D.** (2011). *Calculus of Variations and Optimal Control Theory*. Princeton.
   - Capítulo 7: Pontryagin Maximum Principle

### Artigos sobre Controle de Epidemias

4. **Hansen, E., Day, T.** (2011). "Optimal control of epidemics with limited resources". *Journal of Mathematical Biology*, 62(3), 423-451.

5. **Bolzoni, L., Bonacini, E., Soresina, C., Groppi, M.** (2017). "Optimal control of epidemic size and duration with limited resources". *Mathematical Biosciences*, 315, 108269.

6. **Kantner, M., Koprucki, T.** (2020). "Beyond just flattening the curve: Optimal control of epidemics with purely non-pharmaceutical interventions". *Journal of Mathematics in Industry*, 10(1), 23.

### Software

7. **SciPy** - `scipy.integrate.solve_bvp` para TPBVP
8. **PyOpt** - Biblioteca de otimização em Python
9. **CasADi** - Framework de otimização simbólica

---

## 11. Conclusão

A introdução de **controle ótimo** posiciona este trabalho claramente dentro da área de **Engenharia Elétrica - Controle e Automação**, utilizando ferramentas matemáticas avançadas (Princípio do Máximo de Pontryagin, cálculo variacional) para resolver um problema de otimização dinâmica com restrições.

Este plano fornece uma base sólida para implementação completa e demonstra como epidemiologia pode ser tratada como um **problema de controle de sistemas dinâmicos**, área central da Engenharia de Controle.
