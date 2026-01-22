# Modelo SIDARTHE para COVID-19

## Introdução

O modelo SIDARTHE (Susceptible-Infected-Diagnosed-Ailing-Recognized-Threatened-Healed-Extinct) foi desenvolvido especificamente para modelar a pandemia de COVID-19, publicado por Giordano et al. (2020) na revista Nature Medicine.

## Referência Original

**Giordano, G., Blanchini, F., Bruno, R., Colaneri, P., Di Filippo, A., Di Matteo, A., & Colaneri, M. (2020).**
*Modelling the COVID-19 epidemic and implementation of population-wide interventions in Italy.*
**Nature Medicine, 26(6), 855-860.**
DOI: [10.1038/s41591-020-0883-7](https://doi.org/10.1038/s41591-020-0883-7)

---

## Compartimentos

O modelo divide a população em **8 compartimentos**:

| Símbolo | Nome | Descrição |
|---------|------|-----------|
| **S** | Susceptible | Indivíduos suscetíveis à infecção |
| **I** | Infected | Infectados **assintomáticos não detectados** |
| **D** | Diagnosed | Infectados **assintomáticos detectados** (testados positivos) |
| **A** | Ailing | Infectados **sintomáticos não detectados** |
| **R** | Recognized | Infectados **sintomáticos detectados** |
| **T** | Threatened | Casos **graves** (hospitalizados/UTI) |
| **H** | Healed | **Curados** (imunes) |
| **E** | Extinct | **Mortos** |

### Diferenças em relação ao SEIR

- **SEIR**: 4 compartimentos (S, E, I, R)
- **SIDARTHE**: 8 compartimentos, distinguindo:
  - Assintomáticos vs sintomáticos
  - Detectados vs não detectados
  - Casos graves separados
  - Mortes explícitas

---

## Equações Diferenciais

O modelo é descrito pelo seguinte sistema de EDOs:

```
dS/dt = -S · [α(I+D) + β(A+R)] / N

dI/dt = S · [α(I+D) + β(A+R)] / N - (ε + ζ + λ)I

dD/dt = εI - (λ + ρ)D

dA/dt = ζI - (η + μ)A

dR/dt = ηA - (θ + κ + ρ)R

dT/dt = θD + κR - (ν + τ)T

dH/dt = λI + ρD + νT + μA + ρR

dE/dt = τT
```

**Conservação da população**: S + I + D + A + R + T + H + E = N

---

## Parâmetros (12 no total)

### Transmissão

- **α**: Taxa de transmissão por infectados **assintomáticos** (I, D)
- **β**: Taxa de transmissão por infectados **sintomáticos** (A, R)

*Tipicamente: β > α (sintomáticos transmitem mais)*

### Progressão da doença

- **ε**: Taxa de detecção de assintomáticos (I → D)
- **ζ**: Taxa de desenvolvimento de sintomas (I → A)
- **η**: Taxa de detecção de sintomáticos (A → R)

### Desfechos (cura)

- **λ**: Taxa de cura de infectados não detectados (I → H)
- **μ**: Taxa de cura de sintomáticos não detectados (A → H)
- **ρ**: Taxa de cura de detectados (D → H, R → H)

### Agravamento

- **θ**: Taxa de agravamento de assintomáticos detectados (D → T)
- **κ**: Taxa de agravamento de sintomáticos detectados (R → T)

### Desfechos de casos graves

- **ν**: Taxa de cura de casos graves (T → H)
- **τ**: Taxa de mortalidade (T → E)

---

## Diagrama de Transições

```
           α(I+D) + β(A+R)
    S  ────────────────────────→  I, A

    I  ──ε──→  D  (diagnóstico)
    │
    └──ζ──→  A  (sintomas)
    │
    └──λ──→  H  (cura)

    A  ──η──→  R  (diagnóstico)
    │
    └──μ──→  H  (cura)

    D  ──θ──→  T  (agravamento)
    │
    ├──ρ──→  H  (cura)
    └──λ──→  H  (cura alternativa)

    R  ──κ──→  T  (agravamento)
    │
    └──ρ──→  H  (cura)

    T  ──ν──→  H  (cura UTI)
    │
    └──τ──→  E  (morte)
```

---

## Número Reprodutivo Básico (R₀)

Para SIDARTHE, R₀ é calculado pela **matriz de próxima geração**.

**Aproximação simplificada**:
```
R₀ ≈ (α + β) / γ_avg
```

Onde γ_avg é a taxa média de remoção dos compartimentos infecciosos.

**Fórmula exata** (complexa):
```
R₀ = λ_max(FV⁻¹)
```
Onde F = matriz de infecção, V = matriz de transição.

Para derivação completa, ver:
- van den Driessche & Watmough (2002)
- Giordano et al. (2020) material suplementar

---

## Aplicação ao COVID-19

### Dados observáveis que podemos usar

1. **Casos confirmados totais**: D + R + T + H + E
2. **Casos ativos**: D + R + T
3. **Casos graves (hospitalizados)**: T
4. **Mortes acumuladas**: E
5. **Recuperados**: H

### Vantagem

O modelo pode ser ajustado **diretamente** a esses dados reais!

### Limitações do SEIR

- SEIR só modela I (infectados totais)
- Não distingue casos detectados vs não detectados
- Dificulta ajuste a dados reais de COVID-19

---

## Exemplo de Valores de Parâmetros

### Itália (Giordano et al. 2020)

| Parâmetro | Valor | Unidade | Descrição |
|-----------|-------|---------|-----------|
| α | 0.570 | dia⁻¹ | Transmissão assintomáticos |
| β | 0.011 | dia⁻¹ | Transmissão sintomáticos |
| ε | 0.171 | dia⁻¹ | Detecção assintomáticos |
| ζ | 0.125 | dia⁻¹ | Desenvolvimento sintomas |
| η | 0.125 | dia⁻¹ | Detecção sintomáticos |
| λ | 0.034 | dia⁻¹ | Cura não detectados |
| μ | 0.017 | dia⁻¹ | Cura sintomáticos não detect. |
| ρ | 0.017 | dia⁻¹ | Cura detectados |
| θ | 0.371 | dia⁻¹ | Agravamento D→T |
| κ | 0.017 | dia⁻¹ | Agravamento R→T |
| ν | 0.125 | dia⁻¹ | Cura casos graves |
| τ | 0.010 | dia⁻¹ | Mortalidade |

**Nota**: Estes valores são para a Itália em 2020. Devem ser **re-identificados para o Brasil**!

---

## Comparação SEIR vs SIDARTHE

| Aspecto | SEIR | SIDARTHE |
|---------|------|----------|
| Compartimentos | 4 | 8 |
| Parâmetros | 3 (β, σ, γ) | 12 |
| Assintomáticos | Não distingue | Sim (I, D) |
| Sintomáticos | Não distingue | Sim (A, R) |
| Detecção | Não modela | Explícita |
| Hospitalizações | Não modela | Sim (T) |
| Mortes | Não explícitas | Explícitas (E) |
| Ajuste a dados reais | Difícil | Direto |
| Complexidade | Baixa | Alta |

---

## Implementação em Python

Ver: [src/models/sidarthe.py](../src/models/sidarthe.py)

### Exemplo de uso

```python
from src.models.sidarthe import SIDARTHEModel
import numpy as np

# Definir parâmetros (exemplo da Itália)
params = {
    'alpha': 0.570, 'beta': 0.011, 'epsilon': 0.171,
    'zeta': 0.125, 'eta': 0.125, 'lambda_': 0.034,
    'mu': 0.017, 'rho': 0.017, 'theta': 0.371,
    'kappa': 0.017, 'nu': 0.125, 'tau': 0.010
}

# Criar modelo
model = SIDARTHEModel(params, N=60e6)  # Itália ~60M

# Condições iniciais [S, I, D, A, R, T, H, E]
y0 = [59.9e6, 200, 20, 20, 0, 0, 0, 5]

# Simular 100 dias
t = np.linspace(0, 100, 101)
sol = model.simulate(y0, (0, 100), t)

# Calcular observáveis
obs = model.compute_observables(sol)

# Acessar resultados
casos_confirmados = obs['confirmed']
mortes = obs['deaths']
hospitalizados = obs['hospitalized']
```

---

## Interpretação dos Compartimentos

### Fluxo típico de um indivíduo

1. **S → I**: Suscetível infectado (assintomático não detectado)
2. **I → A**: Desenvolve sintomas (ainda não detectado)
3. **A → R**: É testado e diagnosticado (sintomático detectado)
4. **R → T**: Casos graves vão para hospital/UTI
5. **T → H** ou **T → E**: Cura ou morte

### Rotas alternativas

- **I → D**: Assintomático detectado (por testagem em massa)
- **I → H**: Assintomático cura sem ser detectado
- **A → H**: Sintomático leve cura sem ser detectado
- **D → T**: Assintomático detectado agrava (raro)

---

## Questões de Modelagem

### Por que α ≠ β?

- **α** (assintomáticos): Transmissão menor, mas **mais tempo** infeccioso
- **β** (sintomáticos): Transmissão potencialmente maior, mas pessoas se isolam

### Por que θ > κ na prática?

- **θ** (D→T): Assintomáticos detectados são testados por política ativa (pode incluir casos leves)
- **κ** (R→T): Sintomáticos detectados já estão monitorados

Na realidade, κ > θ (sintomáticos agravam mais). Os valores dependem da **política de testagem**!

### Limitações do modelo

1. **Não modela reinfecção** (H é permanentemente imune)
2. **Parâmetros constantes** (não captura mudanças em lockdown, variantes)
3. **População homogênea** (não considera idade, comorbidades)
4. **Sem vacinação** (pode ser estendido)

---

## Extensões Possíveis

### SIDARTHE com Vacinação

Adicionar fluxo: **S → H** com taxa u(t) (vacinação)

```
dS/dt = -S·[α(I+D) + β(A+R)]/N - u(t)
dH/dt = λI + ρD + νT + μA + ρR + u(t)
```

### SIDARTHE com Lockdown

Modelar α e β como funções do tempo: α(t), β(t)

Durante lockdown: α(t) = α₀·(1 - u₁(t))

### SIDARTHE Multi-Age

Criar 8 compartimentos para cada faixa etária, com matriz de contato.

---

## Referências Adicionais

1. **Artigo original**:
   - Giordano et al. (2020), Nature Medicine

2. **Implementações**:
   - Código original: https://github.com/fprc/COVID-19

3. **Análise matemática**:
   - Brauer, F., Castillo-Chavez, C., & Feng, Z. (2019).
     *Mathematical Models in Epidemiology*. Springer.

4. **Next-generation matrix**:
   - van den Driessche, P., & Watmough, J. (2002).
     *Reproduction numbers and sub-threshold endemic equilibria for compartmental models of disease transmission*.
     Mathematical Biosciences, 180(1-2), 29-48.

5. **Identifiability analysis**:
   - Imron, M. A., Gerardo, B. D., & Lazuardi, L. (2021).
     *Identifiability analysis and parameter estimation of SIDARTHE model for COVID-19*.

---

## Ver também

- [Identificação de Parâmetros SIDARTHE](identificacao_sidarthe.md)
- [Modelo SEIR](modelo_seir.md)
- [Controle Ótimo](plano_controle_otimo.md)
