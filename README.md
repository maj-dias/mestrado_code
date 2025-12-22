# Identificação de Sistemas Epidemiológicos

Projeto de mestrado focado em identificação de parâmetros de modelos epidemiológicos usando dados reais de COVID-19 do Brasil.

## 📋 Descrição

Este projeto implementa um sistema completo de identificação de parâmetros para o modelo SIR (Susceptible-Infected-Recovered) usando otimização por mínimos quadrados não-lineares. O sistema estima os parâmetros β (taxa de transmissão) e γ (taxa de recuperação) a partir de dados reais da pandemia de COVID-19 no Brasil.

## 🎯 Funcionalidades

- ✅ Carregamento e processamento de dados COVID-19 do Brasil (2020-2025)
- ✅ Modelo SIR com equações diferenciais
- ✅ Identificação de parâmetros por mínimos quadrados
- ✅ Visualização de resultados com gráficos de diagnóstico
- ✅ Métricas de qualidade de ajuste (R², RMSE, MAPE)
- ✅ Configuração flexível via YAML

## 📁 Estrutura do Projeto

```
mestrado/
├── src/
│   ├── models/          # Modelos epidemiológicos (SIR, SEIR)
│   ├── identification/  # Algoritmos de identificação
│   ├── utils/           # Utilitários de carregamento de dados
│   └── plots/           # Funções de visualização
├── scripts/             # Scripts executáveis
│   └── identification_sir.py  # Script principal
├── data/
│   └── raw/            # Dados COVID-19 (CSVs)
├── config/
│   └── default.yaml    # Configuração padrão
├── results/            # Resultados (criado automaticamente)
│   ├── parameters/     # Parâmetros estimados (JSON)
│   └── figures/        # Gráficos gerados
└── requirements.txt    # Dependências Python

## 🚀 Instalação

### 1. Clone o repositório

```bash
cd c:\git_projects\mestrado
```

### 2. Crie um ambiente virtual (recomendado)

```bash
python -m venv venv
```

### 3. Ative o ambiente virtual

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Instale as dependências

```bash
pip install -r requirements.txt
```

## 📊 Uso

### Executar Identificação de Parâmetros

```bash
python scripts/identification_sir.py
```

O script irá:
1. Carregar dados COVID-19 de `data/raw/`
2. Preparar compartimentos SIR (S, I, R)
3. Estimar parâmetros β e γ
4. Salvar resultados em `results/`
5. Gerar gráficos de diagnóstico

### Configuração

Edite `config/default.yaml` para ajustar:

- **Período de dados**: `start_date` e `end_date`
- **Chutes iniciais**: `initial_guess.beta` e `initial_guess.gamma`
- **Limites dos parâmetros**: `bounds.beta` e `bounds.gamma`
- **Algoritmo de otimização**: `optimizer.algorithm` (L-BFGS-B, TNC, SLSQP)

## 📈 Resultados

Após a execução, os resultados são salvos em:

### Parâmetros Estimados
`results/parameters/sir_params.json`
```json
{
  "beta": 0.342156,
  "gamma": 0.085234,
  "R0": 4.0145,
  "infectious_period_days": 11.7,
  "metrics": {
    "r2": 0.9234,
    "rmse": 12543.67
  }
}
```

### Gráficos
- `results/figures/sir_fit.png` - Ajuste do modelo aos dados
- `results/figures/compartments.png` - Evolução de S, I, R
- `results/figures/residuals.png` - Análise de resíduos

## 🧪 Modelo SIR

O modelo SIR descreve a dinâmica epidemiológica através de três compartimentos:

**Equações:**
```
dS/dt = -β * S * I / N
dI/dt = β * S * I / N - γ * I
dR/dt = γ * I
```

**Parâmetros:**
- **β (beta)**: Taxa de transmissão (contatos efetivos por dia)
- **γ (gamma)**: Taxa de recuperação (1 / duração da infecção)
- **R₀**: Número básico de reprodução = β / γ

**Interpretação de R₀:**
- R₀ > 1: Epidemia em crescimento
- R₀ = 1: Epidemia estável
- R₀ < 1: Epidemia em declínio

## 📚 Dados

Os dados utilizados são do Ministério da Saúde do Brasil:
- Fonte: Painel COVID-19 Brasil
- Período: 2020-2025
- Granularidade: Nacional (Brasil)
- Atualização: Dados até setembro de 2025

## 🔧 Desenvolvimento

### Adicionar Novo Modelo

1. Crie o arquivo em `src/models/novo_modelo.py`
2. Implemente a classe seguindo o padrão de `SIRModel`
3. Adicione ao `src/models/__init__.py`

### Adicionar Novo Método de Identificação

1. Crie o arquivo em `src/identification/novo_metodo.py`
2. Implemente a função de identificação
3. Adicione ao `src/identification/__init__.py`

## 📖 Referências

- Kermack, W. O., & McKendrick, A. G. (1927). A contribution to the mathematical theory of epidemics.
- Martcheva, M. (2015). An Introduction to Mathematical Epidemiology.

## 📝 Licença

Projeto acadêmico - Mestrado

## 👤 Autor

Projeto desenvolvido como parte do programa de mestrado em [Área do Mestrado]
