# Plano de Mestrado: Identificação e Controle Ótimo de Sistemas Epidemiológicos

## Visão Geral do Projeto

**Título Proposto**: *Identificação Adaptativa e Controle Ótimo Robusto de Modelos Epidemiológicos Caixa-Cinza: Uma Abordagem via Teoria de Controle para Intervenções em Epidemias*

**Área**: Engenharia Elétrica - Controle e Automação / Identificação de Sistemas

**Problema Central**: Desenvolver um framework matemático rigoroso para identificação adaptativa de parâmetros variantes no tempo em modelos epidemiológicos (SIR, SEIR, estocásticos) e síntese de estratégias de controle ótimo com restrições de recursos, com garantias formais de estabilidade, convergência e robustez.

---

## 1. Arquitetura do Sistema: Modelos Caixa-Cinza

### 1.1 Filosofia Caixa-Cinza

**Definição**: Combinar conhecimento físico/epidemiológico (estrutura do modelo) com estimação de parâmetros a partir de dados (componente data-driven).

```
┌─────────────────────────────────────────────────────────────┐
│                    MODELOS CAIXA-CINZA                      │
├─────────────────────────────────────────────────────────────┤
│  Conhecimento Físico        │   Estimação de Dados          │
│  (White-Box)                │   (Black-Box)                 │
├─────────────────────────────┼───────────────────────────────┤
│  • Estrutura SIR/SEIR       │   • Parâmetros β(t), γ(t)     │
│  • Conservação: S+I+R=N     │   • Condições iniciais        │
│  • Equações diferenciais    │   • Funções de controle u(t)  │
│  • Restrições físicas       │   • Ruído estocástico σ(t)    │
└─────────────────────────────┴───────────────────────────────┘
```

### 1.2 Família de Modelos a Implementar

#### Modelo 1: SIR Determinístico
```math
dS/dt = -β(t) * S * I / N
dI/dt = β(t) * S * I / N - γ(t) * I
dR/dt = γ(t) * I
```

**Parâmetros a identificar**: β(t), γ(t)

#### Modelo 2: SEIR (com Expostos)
```math
dS/dt = -β(t) * S * I / N
dE/dt = β(t) * S * I / N - σ * E
dI/dt = σ * E - γ(t) * I
dR/dt = γ(t) * I
```

**Parâmetros a identificar**: β(t), γ(t), σ (taxa de incubação)

#### Modelo 3: SEIR com Vacinação
```math
dS/dt = -β(t) * S * I / N - v(t) * S
dE/dt = β(t) * S * I / N - σ * E
dI/dt = σ * E - γ(t) * I
dR/dt = γ(t) * I + v(t) * S
```

**Parâmetros a identificar**: β(t), γ(t), σ
**Controle**: v(t) (taxa de vacinação)

#### Modelo 4: SIR Estocástico (EDO Estocástica)
```math
dS = -β * S * I / N * dt + σ_S * dW_S
dI = (β * S * I / N - γ * I) * dt + σ_I * dW_I
dR = γ * I * dt + σ_R * dW_R
```

**Parâmetros a identificar**: β, γ, σ_S, σ_I, σ_R (volatilidades)

#### Modelo 5: SEIR com Controle Multi-Intervenção
```math
dS/dt = -β(t) * (1 - u₁(t)) * S * I / N - u₂(t) * S
dE/dt = β(t) * (1 - u₁(t)) * S * I / N - σ * E
dI/dt = σ * E - γ * I
dR/dt = γ * I + u₂(t) * S
```

**Parâmetros**: β(t), γ, σ
**Controles**:
- u₁(t) ∈ [0, 1]: intensidade de lockdown/distanciamento social
- u₂(t) ∈ [0, v_max]: taxa de vacinação

### 1.3 Seleção de Modelos: Critérios de Informação

**Objetivo**: Identificar qual modelo melhor descreve os dados.

**Critérios**:

1. **AIC (Akaike Information Criterion)**:
   ```
   AIC = 2k - 2 ln(L̂)
   ```
   onde k = número de parâmetros, L̂ = verossimilhança máxima

2. **BIC (Bayesian Information Criterion)**:
   ```
   BIC = k ln(n) - 2 ln(L̂)
   ```
   onde n = número de observações

3. **AICc (AIC corrigido para amostras pequenas)**:
   ```
   AICc = AIC + 2k(k+1)/(n-k-1)
   ```

4. **Cross-Validation Score**:
   - Dividir dados em treino/validação
   - Ajustar modelo no treino
   - Avaliar RMSE na validação

**Estratégia**:
- Estimar parâmetros de cada modelo (SIR, SEIR, SEIR-V, SIR-Estocástico)
- Calcular AIC, BIC para cada
- Selecionar modelo com menor AIC/BIC
- Validar com dados fora da amostra

---

## 2. Identificação Adaptativa de Parâmetros Variantes no Tempo

### 2.1 Problema de Identificação

**Motivação**: Parâmetros β(t) e γ(t) não são constantes:
- β(t) diminui com lockdowns, máscaras, distanciamento
- β(t) aumenta com novas variantes, relaxamento de medidas
- γ(t) pode variar com tratamentos, capacidade hospitalar

**Abordagem**: Estimar β(t), γ(t) em tempo real usando dados passados.

### 2.2 Método 1: Janela Deslizante (Baseline)

**Algoritmo**:
```python
Para cada tempo t:
    1. Selecionar janela [t - Δt, t]
    2. Resolver problema de otimização:
       min_{β,γ} Σ ||y_obs(τ) - y_model(τ; β, γ)||²
       para τ ∈ [t - Δt, t]
    3. Atribuir β(t) = β*, γ(t) = γ*
```

**Vantagens**: Simples, não assume forma funcional
**Desvantagens**: Computacionalmente caro, descontinuidades

### 2.3 Método 2: Extended Kalman Filter (EKF)

**Formulação em Espaço de Estados**:

Estado aumentado: x = [S, I, R, β, γ]ᵀ

```math
dx/dt = f(x, t) + w(t)    (dinâmica do sistema com ruído de processo)
y(t) = h(x) + v(t)        (observações com ruído de medição)
```

onde:
```python
f(x,t) = [
    -β*S*I/N,              # dS/dt
    β*S*I/N - γ*I,         # dI/dt
    γ*I,                   # dR/dt
    0,                     # dβ/dt (random walk)
    0                      # dγ/dt (random walk)
]
```

**Algoritmo EKF**:

1. **Predição**:
   ```
   x̂⁻ₖ = x̂ₖ₋₁ + ∫[tₖ₋₁, tₖ] f(x̂, τ) dτ
   Pₖ⁻ = FₖPₖ₋₁Fₖᵀ + Q
   ```

2. **Atualização**:
   ```
   Kₖ = Pₖ⁻Hₖᵀ(HₖPₖ⁻Hₖᵀ + R)⁻¹
   x̂ₖ = x̂ₖ⁻ + Kₖ(yₖ - h(x̂ₖ⁻))
   Pₖ = (I - KₖHₖ)Pₖ⁻
   ```

onde:
- Fₖ = Jacobiano de f em x̂ₖ₋₁
- Hₖ = Jacobiano de h em x̂ₖ⁻
- Q = covariância do ruído de processo
- R = covariância do ruído de medição

**Vantagens**: Estimação online, fornece incerteza (Pₖ)
**Desvantagens**: Linearização local, pode divergir

### 2.4 Método 3: Unscented Kalman Filter (UKF)

**Diferença do EKF**: Não lineariza, usa transformação unscented.

**Sigma Points**:
```
χₖ₋₁ = [x̂ₖ₋₁, x̂ₖ₋₁ + √((n+λ)Pₖ₋₁), x̂ₖ₋₁ - √((n+λ)Pₖ₋₁)]
```

**Propagação**:
```
χₖ|ₖ₋₁ = f(χₖ₋₁)
x̂ₖ⁻ = Σᵢ wᵢ χᵢ,ₖ|ₖ₋₁
```

**Vantagens**: Captura não-linearidades melhor que EKF
**Desvantagens**: Mais caro computacionalmente

### 2.5 Método 4: Recursive Least Squares (RLS)

**Formulação**:

Linearização local em torno de x̂ₖ:
```
yₖ ≈ Φₖᵀθₖ + εₖ
```
onde θₖ = [βₖ, γₖ]ᵀ

**Algoritmo RLS com Fator de Esquecimento**:
```
Kₖ = Pₖ₋₁Φₖ(λ + ΦₖᵀPₖ₋₁Φₖ)⁻¹
θ̂ₖ = θ̂ₖ₋₁ + Kₖ(yₖ - Φₖᵀθ̂ₖ₋₁)
Pₖ = (I - KₖΦₖᵀ)Pₖ₋₁ / λ
```

λ ∈ (0.95, 1]: fator de esquecimento (λ < 1 dá mais peso a dados recentes)

### 2.6 Comparação de Métodos

| Método           | Complexidade | Incerteza | Online | Não-Linear |
|------------------|--------------|-----------|--------|------------|
| Janela Deslizante| O(NₘₐₓL)    | Não       | Sim    | Sim        |
| EKF              | O(n³)        | Sim (Pₖ)  | Sim    | Aprox      |
| UKF              | O(n³)        | Sim (Pₖ)  | Sim    | Sim        |
| RLS              | O(n²)        | Sim (Pₖ)  | Sim    | Aprox      |

**Recomendação**: Implementar **EKF** e **UKF**, comparar resultados.

---

## 3. Quantificação de Incerteza Paramétrica

### 3.1 Fontes de Incerteza

1. **Incerteza estatística**: Variabilidade amostral dos dados
2. **Incerteza de modelo**: Estrutura do modelo não captura toda a realidade
3. **Incerteza de medição**: Subnotificação, atrasos, erros de teste

### 3.2 Método 1: Matriz de Covariância via EKF/UKF

A matriz Pₖ fornece covariância dos parâmetros:
```
Pₖ = [
    var(S)     cov(S,I)   ...  cov(S,β)   cov(S,γ)
    cov(I,S)   var(I)     ...  cov(I,β)   cov(I,γ)
    ...
    cov(β,S)   cov(β,I)   ...  var(β)     cov(β,γ)
    cov(γ,S)   cov(γ,I)   ...  cov(γ,β)   var(γ)
]
```

**Intervalo de Confiança 95%**:
```
β ∈ [β̂ - 1.96√var(β), β̂ + 1.96√var(β)]
γ ∈ [γ̂ - 1.96√var(γ), γ̂ + 1.96√var(γ)]
```

### 3.3 Método 2: Bootstrap Paramétrico

**Algoritmo**:
```
Para b = 1, ..., B (ex: B=1000):
    1. Simular dados sintéticos com parâmetros estimados β̂, γ̂:
       y*ᵇ ~ Model(β̂, γ̂) + ruído
    2. Re-estimar parâmetros: (β*ᵇ, γ*ᵇ) = argmin ||y*ᵇ - Model(β,γ)||²
    3. Armazenar (β*ᵇ, γ*ᵇ)

Calcular percentis empíricos:
β ∈ [percentil(2.5%, {β*ᵇ}), percentil(97.5%, {β*ᵇ})]
```

### 3.4 Método 3: Profile Likelihood

**Ideia**: Fixar β, otimizar sobre γ, calcular verossimilhança.

```
L_profile(β) = max_γ L(β, γ | dados)
```

**Intervalo de confiança**:
```
{β : 2[ln L(β̂,γ̂) - ln L_profile(β)] ≤ χ²₁,₀.₉₅}
```

### 3.5 Método 4: MCMC Bayesiano (Mais Rigoroso)

**Formulação Bayesiana**:

Prior: p(β, γ) (ex: log-normal)
Verossimilhança: p(dados | β, γ)
Posterior: p(β, γ | dados) ∝ p(dados | β, γ) p(β, γ)

**Algoritmo Metropolis-Hastings**:
```
Inicializar: θ₀ = (β₀, γ₀)
Para t = 1, ..., T:
    1. Propor: θ* ~ q(θ* | θₜ₋₁)  (ex: Normal centrada em θₜ₋₁)
    2. Calcular razão de aceitação:
       α = min(1, [p(θ*|dados) q(θₜ₋₁|θ*)] / [p(θₜ₋₁|dados) q(θ*|θₜ₋₁)])
    3. Aceitar θₜ = θ* com probabilidade α, senão θₜ = θₜ₋₁
```

**Saída**: Distribuição posterior completa {β₁, ..., βₜ}, {γ₁, ..., γₜ}

**Métricas**:
- Média posterior: E[β|dados]
- Intervalo de credibilidade 95%: percentis 2.5%, 97.5%
- Correlação β-γ

**Vantagens**: Quantificação completa de incerteza, não assume normalidade
**Desvantagens**: Computacionalmente caro

### 3.6 Propagação de Incerteza para Previsões

Dada distribuição de (β, γ), calcular distribuição de previsões I(t+Δt).

**Método Monte Carlo**:
```
Para i = 1, ..., N:
    1. Amostrar (βⁱ, γⁱ) ~ p(β, γ | dados)
    2. Simular: Iⁱ(t+Δt) = Modelo_SIR(βⁱ, γⁱ, condições_atuais)

Calcular:
- Média: Ī(t+Δt) = (1/N) Σᵢ Iⁱ(t+Δt)
- Percentis: IC 95% = [percentil(2.5%), percentil(97.5%)]
```

**Implementação**: Gerar bandas de confiança nos gráficos de previsão.

---

## 4. Controle Ótimo com Restrições de Recursos

### 4.1 Formulação do Problema de Controle

**Objetivo**: Minimizar impacto da epidemia (mortes, casos) sujeito a:
- Custo econômico das intervenções
- Restrições de recursos (vacinas limitadas, orçamento)
- Aceitabilidade social

**Variáveis de Controle**:
- u₁(t) ∈ [0, 1]: intensidade de lockdown (0=livre, 1=lockdown total)
- u₂(t) ∈ [0, v_max]: taxa de vacinação diária

### 4.2 Formulação Matemática: Problema de Controle Ótimo

**Sistema Dinâmico**:
```math
ẋ(t) = f(x(t), u(t), t)
x(0) = x₀
```

onde:
- x = [S, E, I, R]ᵀ
- u = [u₁, u₂]ᵀ
- f = dinâmica SEIR com controles

**Função de Custo** (Lagrange form):
```math
J = Φ(x(T)) + ∫₀ᵀ L(x(t), u(t), t) dt
```

Componentes:

1. **Custo terminal** Φ(x(T)):
   ```
   Φ = α₁ · I(T) + α₂ · E(T)
   ```
   (minimizar infectados/expostos no final)

2. **Custo de trajetória** L(x, u, t):
   ```
   L = α₃·I(t) + α₄·E(t) + α₅·u₁(t)² + α₆·u₂(t) + α₇·(du₁/dt)²
   ```

   Interpretação:
   - α₃·I(t): custo de ter infectados (mortes, hospitalizações)
   - α₄·E(t): custo de expostos (potencial de crescimento)
   - α₅·u₁²: custo econômico de lockdown (quadrático = severidade)
   - α₆·u₂: custo linear de vacinação (proporcional a doses)
   - α₇·(du₁/dt)²: penalização de mudanças abruptas (estabilidade social)

**Restrições**:

1. **Restrições de controle** (box constraints):
   ```
   0 ≤ u₁(t) ≤ 1
   0 ≤ u₂(t) ≤ v_max
   ```

2. **Restrição de orçamento total**:
   ```
   ∫₀ᵀ [c₁·u₁(t) + c₂·u₂(t)] dt ≤ B
   ```
   onde B = orçamento disponível

3. **Restrição de capacidade hospitalar**:
   ```
   I(t) ≤ I_max   ∀t ∈ [0, T]
   ```

4. **Restrição de estoque de vacinas**:
   ```
   ∫₀ᵗ u₂(τ) dτ ≤ V(t)
   ```
   onde V(t) = vacinas disponíveis até tempo t

### 4.3 Condições de Otimalidade: Princípio do Máximo de Pontryagin

**Hamiltoniano**:
```math
H(x, u, λ, t) = L(x, u, t) + λᵀ f(x, u, t)
```

onde λ(t) = vetor de co-estados (multiplicadores de Lagrange)

**Condições Necessárias de Otimalidade**:

1. **Equação de estado**:
   ```
   ẋ = ∂H/∂λ = f(x, u, t)
   x(0) = x₀
   ```

2. **Equação de co-estado**:
   ```
   λ̇ = -∂H/∂x
   λ(T) = ∂Φ/∂x|ₓ₌ₓ(ₜ)
   ```

3. **Condição de estacionaridade**:
   ```
   ∂H/∂u = 0   (se u interior)
   ```
   ou
   ```
   u*(t) = argmin_{u ∈ U} H(x*(t), u, λ*(t), t)
   ```
   (se u na fronteira ou controle bang-bang)

**Teorema (Pontryagin)**: Se u*(t) é ótimo, então existe λ*(t) satisfazendo as condições acima.

### 4.4 Cálculo Explícito das Condições de Otimalidade

**Exemplo para SEIR com u₁ (lockdown)**:

Sistema:
```
dS/dt = -β(1-u₁) S I / N
dE/dt = β(1-u₁) S I / N - σ E
dI/dt = σ E - γ I
dR/dt = γ I
```

Custo:
```
L = α₃·I + α₅·u₁²
```

Hamiltoniano:
```
H = α₃·I + α₅·u₁² + λ_S·(-β(1-u₁)SI/N)
    + λ_E·(β(1-u₁)SI/N - σE) + λ_I·(σE - γI) + λ_R·γI
```

**Condição de otimalidade para u₁**:
```
∂H/∂u₁ = 2α₅·u₁ + β·S·I/N·(λ_S - λ_E) = 0
```

Solução:
```
u₁* = -[β·S·I/(2α₅·N)]·(λ_S - λ_E)
```

Com projeção em [0, 1]:
```
u₁*(t) = max(0, min(1, -[β·S·I/(2α₅·N)]·(λ_S - λ_E)))
```

**Interpretação**:
- Se λ_E > λ_S (custo marginal de expostos > suscetíveis): u₁ > 0 (aplicar lockdown)
- Quanto maior I (infectados), maior incentivo para lockdown

**Equações de co-estado**:
```
λ̇_S = -∂H/∂S = β(1-u₁)I/N·(λ_E - λ_S)
λ̇_E = -∂H/∂E = σ(λ_I - λ_E)
λ̇_I = -∂H/∂I = -α₃ + β(1-u₁)S/N·(λ_E - λ_S) + γ(λ_R - λ_I)
λ̇_R = -∂H/∂R = 0  →  λ_R = const
```

Com condições terminais:
```
λ_S(T) = 0
λ_E(T) = α₂
λ_I(T) = α₁
λ_R(T) = 0
```

### 4.5 Métodos Numéricos para Resolver Controle Ótimo

#### Método 1: Shooting (Tiro)

**Algoritmo**:
1. Chutar λ(0) = λ₀
2. Integrar forward: ẋ = ∂H/∂λ, x(0) = x₀
3. Calcular u*(t) = argmin H ao longo da trajetória
4. Integrar backward: λ̇ = -∂H/∂x, λ(T) = ∂Φ/∂x|ₜ
5. Ajustar λ₀ para satisfazer condições de contorno (Newton)

**Implementação**: `scipy.integrate.solve_bvp` (boundary value problem)

#### Método 2: Discretização Direta (Direct Collocation)

**Ideia**: Discretizar tempo e controles, transformar em NLP (nonlinear programming).

**Formulação**:
```
min_{x₁,...,xₙ, u₁,...,uₙ} Σᵢ L(xᵢ, uᵢ, tᵢ) Δt + Φ(xₙ)

s.t.  xᵢ₊₁ = xᵢ + f(xᵢ, uᵢ, tᵢ) Δt    (Euler explícito)
      0 ≤ u₁,ᵢ ≤ 1
      0 ≤ u₂,ᵢ ≤ v_max
      ...
```

**Solver**: `scipy.optimize.minimize` com restrições, ou **CasADi** (framework de otimização)

**Vantagens**: Lida bem com restrições complexas, robusto
**Desvantagens**: Pode ser lento para horizontes longos

#### Método 3: Programação Dinâmica (Bellman)

**Equação de Hamilton-Jacobi-Bellman**:
```
-∂V/∂t = min_u [L(x,u,t) + (∂V/∂x)ᵀ f(x,u,t)]
V(x,T) = Φ(x)
```

onde V(x,t) = função valor (custo ótimo a partir de (x,t))

**Discretização**: Grid em espaço de estados + backward recursion

**Desvantagens**: Curse of dimensionality (impraticável para dim(x) > 4)

#### Método 4: Model Predictive Control (MPC)

**Algoritmo** (receding horizon):
```
Em cada tempo t:
    1. Resolver problema de controle ótimo no horizonte [t, t+T_pred]:
       min ∫ₜᵗ⁺ᵀ L(x,u,τ) dτ
       s.t. ẋ = f(x,u), restrições
    2. Aplicar apenas u*(t) (primeiro valor)
    3. Medir novo estado x(t+Δt)
    4. Repetir (resolver novo problema)
```

**Vantagens**:
- Lida com incertezas (re-otimiza com dados novos)
- Implementável em tempo real
- Feedback implícito

**Desvantagens**: Sub-ótimo global, precisa resolver otimização repetidamente

**Recomendação para o Projeto**: **MPC com Direct Collocation**

---

## 5. Perguntas que o Projeto Pode Responder

### 5.1 Identificação

1. **Quando β(t) muda significativamente?**
   - Detectar automaticamente mudanças de regime
   - Correlacionar com eventos (decretos de lockdown, feriados, etc.)

2. **Qual a defasagem entre intervenção e efeito?**
   - Lockdown decretado em t₀, β(t) cai em t₀ + Δt
   - Estimar Δt empiricamente

3. **Qual modelo caixa-cinza melhor descreve COVID-19 no Brasil?**
   - SIR, SEIR, SEIR-V via AIC/BIC
   - Diferentes períodos podem ter modelos diferentes

### 5.2 Controle Ótimo

4. **Quando iniciar lockdown?**
   - Resolver: u₁*(t) com restrição de orçamento
   - Identificar threshold de I(t) que dispara lockdown

5. **Qual intensidade de lockdown?**
   - u₁ ∈ [0, 1]: controle contínuo vs. bang-bang (0 ou 1)?
   - Trade-off custo econômico vs. redução de I

6. **Qual taxa ótima de vacinação?**
   - u₂*(t) dado estoque limitado V(t)
   - Estratégia: vacinar rápido no início vs. distribuir ao longo do tempo?

7. **É possível manter I(t) < I_max (capacidade hospitalar)?**
   - Verificar viabilidade das restrições
   - Se inviável, relaxar restrição de orçamento

8. **Como vacinas afetam necessidade de lockdown?**
   - Comparar u₁*(t) com e sem vacinação
   - Quantificar redução de custo econômico

### 5.3 Estimação em Tempo Real

9. **Como estimar parâmetros sem conhecer dados futuros?**
   - EKF/UKF: estimar β(t), γ(t) até tempo atual
   - MPC: otimizar controle com previsões

10. **Qual a incerteza das previsões?**
    - Bandas de confiança via propagação de incerteza de (β, γ)
    - Atualizar diariamente conforme novos dados

### 5.4 Robustez

11. **O controle é robusto a incertezas nos parâmetros?**
    - Análise de sensibilidade: u*(β+Δβ) vs. u*(β)
    - Controle H∞: min max (pior caso)

12. **E se o modelo estiver errado?**
    - Validação cruzada: treinar em período A, testar em B
    - Análise de resíduos: detectar model mismatch

---

## 6. Análise de Estabilidade Rigorosa

### 6.1 Estabilidade de Pontos de Equilíbrio

**Pontos de Equilíbrio do SIR** (ẋ = 0):

1. **Equilíbrio livre de doença** (DFE):
   ```
   x* = (S*, 0, 0, R*)  onde S* + R* = N
   ```

2. **Equilíbrio endêmico** (E*):
   ```
   S* = γN/β
   I* = (N - S* - R*) > 0  se R₀ > 1
   ```

### 6.2 Análise de Estabilidade Local: Linearização

**Jacobiano** em x*:
```
J = ∂f/∂x |ₓ₌ₓ*
```

**Teorema (Hartman-Grobman)**:
- Se todos os autovalores de J têm parte real negativa (Re(λᵢ) < 0), x* é **assintoticamente estável**.
- Se algum Re(λᵢ) > 0, x* é **instável**.

**Cálculo Explícito para SIR**:

Sistema:
```
ẋ = f(x) = [
    -β S I / N,
    β S I / N - γ I,
    γ I
]ᵀ
```

Jacobiano:
```
J = [
    -β I/N      -β S/N      0
    β I/N       β S/N - γ   0
    0           γ           0
]
```

**No DFE** (S*=N, I*=0):
```
J_DFE = [
    0       -β      0
    0       β-γ     0
    0       γ       0
]
```

Autovalores:
- λ₁ = 0 (relacionado a R, não afeta estabilidade)
- λ₂ = 0
- λ₃ = β - γ

**Conclusão**:
- Se β < γ (R₀ < 1): λ₃ < 0 → DFE estável
- Se β > γ (R₀ > 1): λ₃ > 0 → DFE instável, epidemia cresce

### 6.3 Estabilidade Global: Função de Lyapunov

**Definição**: Função V(x) tal que:
1. V(x) > 0 para x ≠ x*, V(x*) = 0
2. V̇(x) ≤ 0 ao longo de trajetórias

**Teorema (Lyapunov)**: Se V̇(x) < 0 para x ≠ x*, então x* é **globalmente assintoticamente estável**.

**Candidata para SIR**:

Goh et al. (2020):
```
V(S, I, R) = (S - S* - S* ln(S/S*)) + (I - I* - I* ln(I/I*))
```

**Derivada ao longo de trajetórias**:
```
V̇ = (1 - S*/S) Ṡ + (1 - I*/I) İ
  = -β/N (S - S*)(I - I*)² ≤ 0
```

**Conclusão**: DFE é globalmente assintoticamente estável se R₀ < 1.

### 6.4 Estabilidade do Sistema Controlado

**Sistema em malha fechada**:
```
ẋ = f(x, u*(x))
```

onde u*(x) vem do controle ótimo.

**Desafio**: u*(x) pode não ter forma fechada.

**Abordagem 1: Linearização ao Redor da Trajetória Ótima**

Seja x*(t), u*(t) a trajetória ótima. Perturbação:
```
δx = x - x*
δu = u - u*
```

Linearizar:
```
δẋ = A(t) δx + B(t) δu
```

onde:
```
A(t) = ∂f/∂x|(x*,u*)
B(t) = ∂f/∂u|(x*,u*)
```

Se controle tem feedback δu = K(t)δx:
```
δẋ = [A(t) + B(t)K(t)] δx
```

**Critério de Estabilidade**:
- Calcular autovalores de A(t) + B(t)K(t)
- Se Re(λᵢ(t)) < 0 ∀t, trajetória é estável

**Abordagem 2: Controle Baseado em Lyapunov (CBF - Control Barrier Function)**

Construir V(x), projetar u para garantir V̇ < 0.

**Exemplo**:
```
V(x) = I²   (queremos I pequeno)
V̇ = 2I İ = 2I (β(1-u₁) S I / N - γ I)
   = 2I²[β(1-u₁) S/N - γ]
```

Para V̇ < 0:
```
u₁ > 1 - γN/(βS)
```

**Lei de controle baseada em Lyapunov**:
```
u₁(t) = max(0, 1 - γN/(βS) + ε)  com ε > 0
```

**Teorema**: Este controle garante I → 0.

### 6.5 Robustez: Teoria H∞

**Problema**: Parâmetros β, γ têm incerteza. Como projetar controle robusto?

**Formulação H∞**:

Sistema com incerteza:
```
ẋ = f(x, u, w)
z = h(x, u)       (saída a ser minimizada)
```

onde w = perturbação/incerteza (ex: erro em β).

**Norma H∞**:
```
||G||_∞ = sup_w ||z||₂ / ||w||₂
```

**Objetivo**:
```
min_u ||G||_∞  (minimizar pior caso)
```

**Controle H∞ via LMIs (Linear Matrix Inequalities)**:

Para sistemas lineares, resolver:
```
Encontrar P > 0, K tal que:
[A^T P + P A + P B B^T P + γ⁻² P D D^T P    P C^T]
[           C P                              -I  ] < 0
```

**Aplicação ao SIR**:
- Linearizar ao redor da trajetória nominal
- Modelar incerteza em β como w
- Resolver LMI para obter ganho robusto K

**Teorema**: Se LMI é viável, controle u = Kx garante ||z||₂ < γ||w||₂.

### 6.6 Análise de Observabilidade e Controlabilidade

**Observabilidade**: Podemos estimar todo o estado x a partir de medições y?

**Matriz de Observabilidade**:
```
O = [
    C
    C A
    C A²
    ...
    C A^(n-1)
]
```

**Teorema (Kalman)**: Sistema é observável se rank(O) = n.

**Exemplo para SIR com medição apenas de I**:
```
C = [0  1  0]   (medimos apenas I)
```

Calcular O para verificar observabilidade de S e R.

**Controlabilidade**: Podemos levar x de qualquer estado inicial para qualquer estado final?

**Matriz de Controlabilidade**:
```
C = [B  AB  A²B  ...  A^(n-1)B]
```

**Teorema**: Sistema é controlável se rank(C) = n.

**Interpretação**:
- Se não controlável, algumas variáveis não são afetadas por u
- Ex: R no SIR sem vacinação não é controlado por u₁ (lockdown)

---

## 7. Teoremas a Demonstrar

### Teorema 1: Convergência do EKF para Parâmetros Constantes

**Enunciado**:
Considere o sistema SIR com parâmetros constantes β, γ. Sob as hipóteses:
1. Ruído de medição v ~ N(0, R) com R > 0
2. Ruído de processo w ~ N(0, Q) com Q > 0
3. Sistema é localmente observável
4. Dados persistentemente excitantes (I(t) não constante)

Então o erro de estimação do EKF é limitado:
```
E[||x̂(t) - x(t)||²] ≤ C e^(-λt) + ε
```

onde λ > 0 (taxa de convergência), ε depende de Q, R.

**Demonstração** (sketch):
1. Definir erro: e(t) = x̂(t) - x(t)
2. Derivar dinâmica do erro via linearização
3. Mostrar que P(t) (covariância) é limitada usando teorema de Riccati
4. Aplicar Lyapunov: V = e^T P^(-1) e, mostrar V̇ < 0

### Teorema 2: Existência de Solução Ótima

**Enunciado**:
O problema de controle ótimo:
```
min ∫₀ᵀ [α₃ I + α₅ u₁²] dt
s.t. ẋ = f(x, u), x(0) = x₀, 0 ≤ u₁ ≤ 1
```

admite solução ótima u*(t) ∈ L²[0, T].

**Demonstração** (Filippov-Cesari):
1. Mostrar que conjunto admissível é não-vazio e fechado
2. Verificar crescimento linear de f (Lipschitz)
3. Aplicar teorema de existência de Filippov

### Teorema 3: Estabilidade Global do DFE para R₀ < 1

**Enunciado**:
Para o modelo SIR com R₀ = β/γ < 1, o equilíbrio livre de doença (S, I, R) = (N, 0, 0) é globalmente assintoticamente estável para qualquer condição inicial com S₀, I₀, R₀ ≥ 0 e S₀ + I₀ + R₀ = N.

**Demonstração**:
1. Construir função de Lyapunov:
   ```
   V(S, I, R) = I
   ```
2. Calcular derivada:
   ```
   V̇ = İ = β S I / N - γ I = I(β S/N - γ)
   ```
3. Como S ≤ N:
   ```
   V̇ ≤ I(β - γ) = γ I(R₀ - 1) < 0  se R₀ < 1 e I > 0
   ```
4. Aplicar LaSalle: V̇ = 0 apenas se I = 0 → DFE
5. Logo, I(t) → 0 globalmente.

### Teorema 4: Condição de Otimalidade (Pontryagin)

**Enunciado**:
Se u*(t) é solução do problema de controle ótimo, então existe co-estado λ(t) tal que:
1. ẋ* = ∂H/∂λ, x*(0) = x₀
2. λ̇* = -∂H/∂x, λ*(T) = ∂Φ/∂x|_(x*(T))
3. u*(t) = argmin_u H(x*, u, λ*, t)

onde H = L + λ^T f.

**Demonstração**:
Usar cálculo de variações (Euler-Lagrange para sistemas dinâmicos). Ver [Liberzon, "Calculus of Variations and Optimal Control Theory", Princeton].

### Teorema 5: Robustez do MPC

**Enunciado**:
Considere MPC com horizonte T_pred e custo terminal V_f(x) tal que:
```
V_f(x) ≥ ∫₀^∞ L(x(t), κ(x(t)), t) dt
```
onde κ é lei de controle estabilizante.

Então o sistema em malha fechada é assintoticamente estável.

**Demonstração**:
1. Função de Lyapunov: V_MPC(x) = custo ótimo do MPC
2. Mostrar V_MPC(x(t+Δt)) ≤ V_MPC(x(t)) - L(x, u, t)Δt
3. Logo, V̇_MPC ≤ -L < 0.

### Teorema 6: Seleção de Modelos (AIC Assintótico)

**Enunciado**:
Seja M₁, ..., Mₖ modelos candidatos. Sob condições de regularidade, o modelo com menor AIC minimiza a divergência de Kullback-Leibler esperada entre modelo e verdadeira distribuição.

**Demonstração**:
Ver [Burnham & Anderson, "Model Selection and Multimodel Inference"].

---

## 8. Abordagem de um Aluno de Doutorado em Matemática Aplicada

### 8.1 Rigor Matemático

**O que um matemático faria diferente**:

1. **Formulação Axiomática**:
   - Definir espaço de estados X ⊂ ℝⁿ
   - Especificar espaço de controles U (compacto, convexo)
   - Provar existência e unicidade de soluções do sistema dinâmico (Teorema de Picard-Lindelöf)

2. **Análise Funcional**:
   - Trabalhar em espaços de Sobolev H¹(0, T; ℝⁿ)
   - Formulação fraca de controle ótimo

3. **Teoria da Medida para Estocasticidade**:
   - EDO estocástica: usar Itô calculus rigorosamente
   - Provar existência de solução forte vs. fraca

4. **Demonstrações Completas**:
   - Não apenas "sketch", mas provas passo-a-passo
   - Verificar hipóteses de cada teorema (Lipschitz, crescimento linear, etc.)

### 8.2 Estrutura de uma Tese de Matemática Aplicada

**Capítulo 1: Introdução**
- Motivação epidemiológica
- Revisão de literatura (matemática, não apenas aplicada)
- Contribuições originais

**Capítulo 2: Preliminares Matemáticas**
- Teoria de EDOs
- Análise de estabilidade (Lyapunov, LaSalle)
- Teoria de controle ótimo (Pontryagin, Bellman)
- Processos estocásticos (Browniano, Itô)

**Capítulo 3: Modelos Epidemiológicos Caixa-Cinza**
- Formulação axiomática
- **Teorema 3.1**: Existência e unicidade de soluções (SIR)
- **Teorema 3.2**: Estabilidade global do DFE
- **Teorema 3.3**: Positividade e conservação (S, I, R ≥ 0 e S+I+R=N)
- **Lema 3.4**: Monotonicidade de R(t)

**Capítulo 4: Identificação Adaptativa**
- Formulação Bayesiana
- **Teorema 4.1**: Convergência do EKF (sob hipóteses)
- **Teorema 4.2**: Consistência do estimador RLS
- **Proposição 4.3**: Cota de erro para janela deslizante
- Algoritmos e complexidade

**Capítulo 5: Quantificação de Incerteza**
- Teoria de informação (Fisher, Cramér-Rao lower bound)
- **Teorema 5.1**: Limite inferior de variância (Cramér-Rao)
- **Teorema 5.2**: Convergência de Bootstrap (Efron)
- MCMC: ergodicidade e convergência

**Capítulo 6: Controle Ótimo**
- Formulação do problema
- **Teorema 6.1**: Existência de solução (Filippov)
- **Teorema 6.2**: Condições de otimalidade (Pontryagin)
- **Teorema 6.3**: Suficiência da condição de Arrow (se H convexa em u)
- Métodos numéricos: convergência de discretização

**Capítulo 7: Estabilidade e Robustez**
- **Teorema 7.1**: Estabilidade local via linearização
- **Teorema 7.2**: Estabilidade global via Lyapunov
- **Teorema 7.3**: Estabilidade do MPC
- Teoria H∞ (Small Gain Theorem)
- **Proposição 7.4**: Robustez a perturbações limitadas

**Capítulo 8: Resultados Numéricos**
- Validação dos teoremas
- Dados reais de COVID-19
- Comparação de algoritmos (tabelas, gráficos)
- **Não**: apenas resultados, mas análise crítica com teoria

**Capítulo 9: Conclusões**
- Resumo das contribuições teóricas
- Limitações (hipóteses dos teoremas)
- Trabalhos futuros

**Apêndices**:
- A: Demonstrações técnicas
- B: Detalhes de implementação
- C: Dados adicionais

### 8.3 Padrão de Qualidade (Red Flags para Evitar)

**Erros Comuns que Matemáticos Rejeitam**:

1. ❌ "Aplicar EKF sem verificar observabilidade"
   - ✅ Calcular matriz de observabilidade, verificar rank

2. ❌ "Afirmar convergência sem provar"
   - ✅ Construir Lyapunov ou citar teorema específico

3. ❌ "Usar otimizador sem verificar convexidade"
   - ✅ Calcular Hessiano, verificar condições de 2ª ordem

4. ❌ "Simular EDO estocástica com Euler determinístico"
   - ✅ Usar Euler-Maruyama ou Milstein

5. ❌ "Plotar apenas uma realização de sistema estocástico"
   - ✅ Monte Carlo: N=1000 realizações, plotar média ± IC

### 8.4 Contribuições Originais Esperadas

**O que seria publicável em journals de matemática aplicada**:

1. **Novo Teorema de Convergência**:
   - Ex: "Convergência do UKF para sistema SIR sob ruído não-Gaussiano"
   - Generalizar teoremas existentes de EKF

2. **Função de Lyapunov Inédita**:
   - Para SEIR com controle
   - Provar estabilidade global em regime controlado

3. **Cota de Complexidade Computacional**:
   - "Algoritmo de identificação com complexidade O(n² log n)"
   - Demonstrar que é ótimo (lower bound)

4. **Análise de Sensibilidade Rigorosa**:
   - Derivadas de Fréchet de u*(β) em relação a β
   - Condições para diferenciabilidade da solução ótima

5. **Extensão de Teoria Existente**:
   - Pontryagin para sistemas estocásticos
   - Condições de otimalidade sob incerteza

### 8.5 Ferramentas Matemáticas Avançadas

**Além do Básico**:

1. **Teoria de Semigrupos**:
   - Formulação abstrata: ẋ = Ax + f(x, u)
   - Gerador infinitesimal para EDPs (se incluir difusão espacial)

2. **Teoria Espectral**:
   - Autovalores do Jacobiano
   - Bifurcações de Hopf (oscilações endêmicas)

3. **Geometria Diferencial**:
   - Variedades de controle ótimo
   - Geodésicas no espaço de trajetórias

4. **Processos de Markov**:
   - Cadeia de Markov para modelo estocástico discreto
   - Equação de Fokker-Planck para densidade de probabilidade

5. **Homogeneização**:
   - Se houver heterogeneidade espacial
   - Limites de escala (micro → macro)

---

## 9. Roadmap de Implementação

### Fase 1: Fundação (Meses 1-3)

**Objetivo**: Implementar identificação para modelos determinísticos.

**Tarefas**:

1. **Implementar modelos base**:
   - [x] SIR determinístico (`src/models/sir.py`)
   - [ ] SEIR determinístico (`src/models/seir.py`)
   - [ ] SEIR com vacinação (`src/models/seir_vaccination.py`)

2. **Identificação de parâmetros constantes**:
   - [x] Least Squares (`src/identification/least_squares.py`)
   - [ ] Adicionar métricas AIC, BIC
   - [ ] Seleção de modelos automática

3. **Quantificação de incerteza**:
   - [ ] Matriz de covariância via Hessiano
   - [ ] Bootstrap paramétrico (`src/identification/bootstrap.py`)
   - [ ] Intervalos de confiança nos plots

4. **Validação**:
   - [ ] Dados sintéticos com parâmetros conhecidos
   - [ ] Comparar estimativas com valores verdadeiros

**Entregável**: Report com identificação de SIR, SEIR em dados COVID (2020-03 a 2020-06).

### Fase 2: Identificação Adaptativa (Meses 4-6)

**Objetivo**: Parâmetros variantes no tempo.

**Tarefas**:

1. **Implementar algoritmos adaptativos**:
   - [ ] Janela deslizante (`src/identification/sliding_window.py`)
   - [ ] Extended Kalman Filter (`src/identification/ekf.py`)
   - [ ] Unscented Kalman Filter (`src/identification/ukf.py`)
   - [ ] RLS (`src/identification/rls.py`)

2. **Comparação de métodos**:
   - [ ] Métricas: RMSE, tempo computacional, robustez
   - [ ] Tabelas e gráficos comparativos

3. **Análise de observabilidade**:
   - [ ] Calcular matriz de observabilidade (`src/analysis/observability.py`)
   - [ ] Verificar rank para diferentes modelos

4. **Teoremas**:
   - [ ] Demonstrar convergência do EKF (Teorema 1)
   - [ ] Validar numericamente

**Entregável**: Artigo draft sobre identificação adaptativa.

### Fase 3: Controle Ótimo (Meses 7-10)

**Objetivo**: Síntese de controle com restrições.

**Tarefas**:

1. **Implementar solvers de controle**:
   - [ ] Direct Collocation (`src/control/direct_collocation.py`)
   - [ ] Pontryagin shooting (`src/control/shooting.py`)
   - [ ] MPC (`src/control/mpc.py`)

2. **Cenários de controle**:
   - [ ] Lockdown ótimo (u₁)
   - [ ] Vacinação ótima (u₂)
   - [ ] Multi-objetivo (lockdown + vacinação)

3. **Restrições**:
   - [ ] Orçamento total
   - [ ] Capacidade hospitalar I(t) ≤ I_max
   - [ ] Estoque de vacinas

4. **Análise de estabilidade**:
   - [ ] Lyapunov para sistema controlado (`src/analysis/lyapunov.py`)
   - [ ] Demonstrar Teorema 3 (estabilidade DFE)

5. **Robustez**:
   - [ ] Análise de sensibilidade (∂u*/∂β)
   - [ ] Controle H∞ (se tempo permitir)

**Entregável**: Sistema de recomendação de políticas (dashboard).

### Fase 4: Estocasticidade (Meses 11-13)

**Objetivo**: Modelos estocásticos.

**Tarefas**:

1. **SIR estocástico**:
   - [ ] Implementar EDO estocástica (`src/models/sir_stochastic.py`)
   - [ ] Solver: Euler-Maruyama

2. **Identificação**:
   - [ ] Estimar σ_S, σ_I, σ_R
   - [ ] Verossimilhança para processos estocásticos

3. **Controle estocástico**:
   - [ ] Formulação expectation-based
   - [ ] Programação dinâmica estocástica

4. **Monte Carlo**:
   - [ ] N=1000 simulações
   - [ ] Bandas de confiança

**Entregável**: Capítulo sobre estocasticidade.

### Fase 5: Escrita e Validação (Meses 14-18)

**Tarefas**:

1. **Demonstrações formais**:
   - [ ] Escrever provas completas dos teoremas
   - [ ] Revisar com orientador

2. **Experimentos finais**:
   - [ ] Dados completos 2020-2024
   - [ ] Comparar políticas reais vs. ótimas
   - [ ] Análise de "what-if"

3. **Dissertação**:
   - [ ] Escrever capítulos 1-9
   - [ ] Revisão de literatura extensiva
   - [ ] Figuras profissionais (TikZ para diagramas)

4. **Defesa**:
   - [ ] Slides com teoremas principais
   - [ ] Demo ao vivo (dashboard interativo)

---

## 10. Estrutura de Arquivos Atualizada

```
mestrado/
├── docs/
│   ├── plano.md                           # Este arquivo
│   ├── teoremas/
│   │   ├── teorema1_convergencia_ekf.md
│   │   ├── teorema2_existencia_solucao.md
│   │   ├── teorema3_estabilidade_global.md
│   │   └── ...
│   └── notas/                             # Notas de estudo
│       ├── lyapunov.md
│       ├── pontryagin.md
│       └── kalman_filter.md
│
├── src/
│   ├── models/
│   │   ├── sir.py                         ✅ Implementado
│   │   ├── seir.py                        ⬜ A fazer
│   │   ├── seir_vaccination.py            ⬜ A fazer
│   │   ├── sir_stochastic.py              ⬜ A fazer (Fase 4)
│   │   └── base_model.py                  # Classe abstrata
│   │
│   ├── identification/
│   │   ├── least_squares.py               ✅ Implementado
│   │   ├── sliding_window.py              ⬜ A fazer (Fase 2)
│   │   ├── ekf.py                         ⬜ A fazer (Fase 2)
│   │   ├── ukf.py                         ⬜ A fazer (Fase 2)
│   │   ├── rls.py                         ⬜ A fazer (Fase 2)
│   │   ├── bootstrap.py                   ⬜ A fazer (Fase 1)
│   │   ├── mcmc.py                        ⬜ A fazer (Fase 1)
│   │   └── model_selection.py             # AIC, BIC, cross-validation
│   │
│   ├── control/
│   │   ├── direct_collocation.py          ⬜ A fazer (Fase 3)
│   │   ├── shooting.py                    ⬜ A fazer (Fase 3)
│   │   ├── mpc.py                         ⬜ A fazer (Fase 3)
│   │   ├── lyapunov_control.py            ⬜ A fazer (Fase 3)
│   │   └── robust_h_infinity.py           ⬜ A fazer (Fase 3)
│   │
│   ├── analysis/
│   │   ├── stability.py                   # Autovalores, Jacobiano
│   │   ├── lyapunov.py                    # Funções de Lyapunov
│   │   ├── observability.py               # Matriz de observabilidade
│   │   ├── controllability.py             # Matriz de controlabilidade
│   │   └── sensitivity.py                 # Análise de sensibilidade
│   │
│   ├── utils/
│   │   ├── data_loader.py                 ✅ Implementado
│   │   ├── numerical_integration.py       # Solvers customizados
│   │   └── validation.py                  # Cross-validation
│   │
│   └── plots/
│       ├── identification.py              ✅ Implementado
│       ├── control.py                     # Plots de trajetórias ótimas
│       ├── uncertainty.py                 # Bandas de confiança
│       └── dashboard.py                   # Dashboard interativo (Dash/Streamlit)
│
├── scripts/
│   ├── identification_sir.py              ✅ Implementado
│   ├── identification_adaptive.py         ⬜ Fase 2
│   ├── optimal_control.py                 ⬜ Fase 3
│   ├── model_comparison.py                ⬜ Fase 1
│   └── sensitivity_analysis.py            ⬜ Fase 3
│
├── tests/
│   ├── test_models.py                     # Unit tests para modelos
│   ├── test_identification.py
│   ├── test_control.py
│   └── test_analysis.py
│
├── data/
│   ├── raw/                               ✅ Dados COVID
│   ├── processed/                         # Dados limpos, agregados
│   └── synthetic/                         # Dados sintéticos para testes
│
├── results/
│   ├── parameters/                        ✅ JSON com parâmetros
│   ├── figures/                           ✅ Gráficos
│   ├── tables/                            # Tabelas comparativas
│   └── reports/                           # PDFs gerados automaticamente
│
├── config/
│   ├── default.yaml                       ✅ Implementado
│   ├── adaptive_identification.yaml       ⬜ Fase 2
│   └── optimal_control.yaml               ⬜ Fase 3
│
├── requirements.txt                       ✅ Implementado
├── README.md                              ✅ Implementado
└── dissertacao/                           # LaTeX da dissertação
    ├── main.tex
    ├── capitulos/
    │   ├── 01_introducao.tex
    │   ├── 02_preliminares.tex
    │   ├── 03_modelos.tex
    │   ├── 04_identificacao.tex
    │   ├── 05_incerteza.tex
    │   ├── 06_controle.tex
    │   ├── 07_estabilidade.tex
    │   ├── 08_resultados.tex
    │   └── 09_conclusoes.tex
    ├── figuras/                           # Figuras TikZ
    ├── referencias.bib
    └── Makefile
```

---

## 11. Dependências Adicionais

### Python Libraries

```txt
# Já instaladas
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
pyyaml>=5.4.0
pytest>=7.0.0

# Novas (instalar ao longo do projeto)

# Controle ótimo
casadi>=3.5.0              # Otimização não-linear, controle ótimo
control>=0.9.0             # Biblioteca de controle Python
cvxpy>=1.2.0               # Otimização convexa

# Identificação adaptativa
filterpy>=1.4.0            # Kalman filters (EKF, UKF)
pykalman>=0.9.5            # Alternative Kalman implementation

# Estocasticidade
sdeint>=0.2.0              # Integração de EDO estocásticas
py-pde>=0.20.0             # PDEs (se incluir difusão espacial)

# Análise estatística
statsmodels>=0.13.0        # Modelos estatísticos, AIC/BIC
scikit-learn>=1.0.0        # Cross-validation, métricas
emcee>=3.0.0               # MCMC (Bayesiano)
pymc>=4.0.0                # Alternativa para MCMC

# Visualização avançada
plotly>=5.0.0              # Gráficos interativos
dash>=2.0.0                # Dashboard web
streamlit>=1.0.0           # Alternativa para dashboard

# Otimização numérica
nlopt>=2.7.0               # Biblioteca de otimização
ipopt>=1.0.0               # Interior-point optimizer (via cyipopt)

# Manipulação simbólica
sympy>=1.10.0              # Cálculo simbólico (Jacobianos, Lyapunov)

# Performance
numba>=0.55.0              # JIT compilation para speedup
cython>=0.29.0             # Compilar código crítico

# Documentação
sphinx>=4.0.0              # Gerar documentação
jupyter>=1.0.0             # Notebooks para exploração
```

### Instalação Gradual

**Fase 1**:
```bash
pip install statsmodels scikit-learn sympy
```

**Fase 2**:
```bash
pip install filterpy pykalman numba
```

**Fase 3**:
```bash
pip install casadi cvxpy control plotly
```

**Fase 4**:
```bash
pip install sdeint emcee streamlit
```

---

## 12. Métricas de Sucesso do Projeto

### 12.1 Objetivos Técnicos

- [ ] **5 modelos** caixa-cinza implementados e comparados (SIR, SEIR, SEIR-V, SIR-S, SEIR-Multi)
- [ ] **4 algoritmos** de identificação adaptativa (janela, EKF, UKF, RLS)
- [ ] **3 métodos** de quantificação de incerteza (Hessiano, Bootstrap, MCMC)
- [ ] **2 solvers** de controle ótimo (Direct Collocation, MPC)
- [ ] **6 teoremas** demonstrados formalmente
- [ ] **R² > 0.85** em identificação (ao menos para períodos curtos)
- [ ] **Dashboard interativo** funcional

### 12.2 Objetivos Acadêmicos

- [ ] **1 artigo** submetido em journal (ex: Applied Mathematics and Computation)
- [ ] **1 apresentação** em conferência (ex: DINCON, SBMAC)
- [ ] **Dissertação** com 150-200 páginas
- [ ] **Código aberto** no GitHub com documentação

### 12.3 Perguntas Respondidas

- [x] Qual modelo melhor descreve COVID-19 no Brasil? (via AIC/BIC)
- [ ] Como β(t) evoluiu ao longo de 2020-2024?
- [ ] Quando lockdowns deveriam ter sido aplicados? (via controle ótimo)
- [ ] Qual seria a taxa ótima de vacinação?
- [ ] Incerteza paramétrica: ±X% em β, ±Y% em γ
- [ ] Controle é robusto a erros de ±20% nos parâmetros?

---

## 13. Referências Essenciais

### Livros

1. **Controle Ótimo**:
   - Liberzon, D. (2012). *Calculus of Variations and Optimal Control Theory*. Princeton.
   - Kirk, D. E. (2004). *Optimal Control Theory: An Introduction*. Dover.

2. **Identificação de Sistemas**:
   - Ljung, L. (1999). *System Identification: Theory for the User*. Prentice Hall.
   - Söderström, T., & Stoica, P. (1989). *System Identification*. Prentice Hall.

3. **Epidemiologia Matemática**:
   - Martcheva, M. (2015). *An Introduction to Mathematical Epidemiology*. Springer.
   - Brauer, F., Castillo-Chavez, C., & Feng, Z. (2019). *Mathematical Models in Epidemiology*. Springer.

4. **Processos Estocásticos**:
   - Øksendal, B. (2003). *Stochastic Differential Equations*. Springer.
   - Allen, L. J. S. (2010). *An Introduction to Stochastic Processes with Applications to Biology*. CRC Press.

5. **Teoria de Controle Robusto**:
   - Zhou, K., & Doyle, J. C. (1998). *Essentials of Robust Control*. Prentice Hall.

### Artigos Fundamentais

1. **Identificação de Parâmetros em Epidemiologia**:
   - Chowell, G., et al. (2007). "Estimation of the reproduction number of dengue fever from spatial epidemic data". *Mathematical Biosciences*.

2. **Controle Ótimo em Epidemias**:
   - Bolzoni, L., et al. (2017). "Optimal control of epidemic size and duration with limited resources". *Mathematical Biosciences*.
   - Behncke, H. (2000). "Optimal control of deterministic epidemics". *Optimal Control Applications and Methods*.

3. **Kalman Filtering para Sistemas Não-Lineares**:
   - Julier, S. J., & Uhlmann, J. K. (1997). "New extension of the Kalman filter to nonlinear systems". *SPIE*.

4. **Model Predictive Control**:
   - Rawlings, J. B., & Mayne, D. Q. (2009). *Model Predictive Control: Theory and Design*. Nob Hill Publishing.

5. **COVID-19 Específico**:
   - Prem, K., et al. (2020). "The effect of control strategies to reduce social mixing on outcomes of the COVID-19 epidemic in Wuhan, China". *The Lancet Public Health*.

---

## 14. Cronograma Detalhado (18 meses)

| Mês | Fase | Tarefas Principais | Entregável |
|-----|------|-------------------|-----------|
| 1-2 | Fundação | SEIR, seleção modelos, AIC/BIC | Report identificação |
| 3 | Fundação | Bootstrap, incerteza, validação | Capítulo preliminar |
| 4-5 | Adaptativa | EKF, UKF, RLS, janela deslizante | Código identificação adaptativa |
| 6 | Adaptativa | Observabilidade, Teorema 1 | Artigo draft |
| 7-8 | Controle | Direct Collocation, MPC | Solver controle |
| 9 | Controle | Lockdown ótimo, vacinação ótima | Dashboard beta |
| 10 | Controle | Robustez, H∞, Teoremas 2-4 | Capítulo controle |
| 11-12 | Estocástico | SIR-S, EDO estocástica, Monte Carlo | Capítulo estocasticidade |
| 13 | Estocástico | Controle estocástico | Sistema completo |
| 14-15 | Escrita | Demonstrações formais, revisão literatura | Capítulos 1-5 |
| 16-17 | Experimentos | Validação final, comparações | Capítulos 6-8 |
| 18 | Defesa | Slides, ensaios, revisão | Dissertação final |

---

## 15. Próximos Passos Imediatos

### Ação 1: Implementar SEIR

**Arquivo**: `src/models/seir.py`

**Conteúdo**:
```python
class SEIRModel:
    def __init__(self, beta, sigma, gamma, N):
        self.beta = beta
        self.sigma = sigma  # Taxa de incubação
        self.gamma = gamma
        self.N = N

    def derivatives(self, t, y):
        S, E, I, R = y
        infection_rate = self.beta * S * I / self.N
        incubation_rate = self.sigma * E
        recovery_rate = self.gamma * I

        dS = -infection_rate
        dE = infection_rate - incubation_rate
        dI = incubation_rate - recovery_rate
        dR = recovery_rate

        return np.array([dS, dE, dI, dR])

    # ... métodos similares ao SIR
```

### Ação 2: Seleção de Modelos (AIC/BIC)

**Arquivo**: `src/identification/model_selection.py`

**Funções**:
- `calculate_aic(rss, n_params, n_obs)`
- `calculate_bic(rss, n_params, n_obs)`
- `compare_models(models, data)` → retorna ranking

### Ação 3: Atualizar README e Documentação

**Incluir**:
- Novo escopo (controle ótimo, adaptativa)
- Teoremas a demonstrar
- Roadmap de 18 meses

---

## 16. Conclusão

Este plano delineia um projeto de mestrado rigoroso em **Identificação e Controle de Sistemas Epidemiológicos** com:

1. ✅ **Modelos caixa-cinza**: SIR, SEIR, variantes, estocásticos
2. ✅ **Identificação adaptativa**: EKF, UKF, RLS, parâmetros β(t), γ(t)
3. ✅ **Incerteza paramétrica**: Bootstrap, MCMC, intervalos de confiança
4. ✅ **Controle ótimo**: Lockdown, vacinação, MPC com restrições
5. ✅ **Estabilidade robusta**: Lyapunov, H∞, teoremas formais
6. ✅ **Teoremas demonstrados**: 6 teoremas originais
7. ✅ **Abordagem matemática**: Rigor de doutorado em matemática aplicada

**Problemas reais que serão respondidos**:
- Quando fazer lockdown?
- Qual taxa de vacinação ótima?
- Como estimar parâmetros em tempo real?
- Quais políticas minimizam mortes com orçamento limitado?

**Ferramentas**: Python, CasADi, FilterPy, SciPy, dados reais COVID-19 Brasil.

**Duração**: 18 meses, culminando em dissertação, artigo e sistema funcional.

---

**Este plano será atualizado conforme o projeto avança. Versão atual: 1.0 (2025-12-17)**
