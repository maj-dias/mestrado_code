# Modelagem Epidemiologica e Controle Otimo - COVID-19

Projeto de mestrado para identificacao de parametros e controle otimo de modelos epidemiologicos (SIR, SEIR, SIDARTHE) aplicados a dados de COVID-19 do Brasil.

## Estrutura do Projeto

```
mestrado_code/
├── config/           # Arquivos de configuracao YAML
├── data/             # Dados brutos COVID-19
├── results/          # Resultados (parametros, graficos)
├── scripts/          # Scripts executaveis
├── src/              # Codigo fonte
│   ├── control/      # Controle otimo (Pontryagin)
│   ├── identification/  # Identificacao de parametros
│   ├── models/       # Modelos epidemiologicos
│   ├── plots/        # Visualizacao
│   └── utils/        # Utilitarios (data loader)
└── requirements.txt
```

## Instalacao

1. Clone o repositorio:
```bash
git clone <url-do-repositorio>
cd mestrado_code
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
```

3. Ative o ambiente virtual:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

4. Instale as dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### 1. Identificacao de Parametros

Os scripts de identificacao estimam os parametros dos modelos epidemiologicos a partir dos dados reais.

**Modelo SIR:**
```bash
python scripts/identification_sir.py
```

**Modelo SEIR:**
```bash
python scripts/identification_seir.py
```

**Modelo SIDARTHE:**
```bash
python scripts/identification_sidarthe.py
```

Os parametros estimados sao salvos em `results/parameters/`.

### 2. Controle Otimo

Apos identificar os parametros, execute o controle otimo usando o Principio do Maximo de Pontryagin:

```bash
python scripts/optimal_control_pontryagin_seir.py
```

Este script:
- Carrega parametros identificados
- Resolve o problema de controle otimo
- Gera trajetorias de controle u1(t) (lockdown) e u2(t) (vacinacao)
- Compara cenarios com e sem controle
- Salva resultados em `results/`

## Configuracao

Os arquivos de configuracao em `config/` permitem ajustar:

- **Dados**: periodo, populacao, suavizacao
- **Identificacao**: chutes iniciais, limites, algoritmo de otimizacao
- **Controle**: pesos da funcao custo, limites de controle, horizonte
- **Saida**: diretorio de resultados, formato de figuras

Exemplo (`config/seir.yaml`):
```yaml
data:
  start_date: "2020-03-01"
  end_date: "2020-12-31"
  population: 210147125

identification:
  initial_guess:
    beta: 0.25
    sigma: 0.2
    gamma: 0.071
```

## Modelos Disponiveis

| Modelo    | Compartimentos | Descricao |
|-----------|----------------|-----------|
| SIR       | S, I, R        | Modelo basico |
| SEIR      | S, E, I, R     | Inclui periodo de exposicao |
| SIDARTHE  | S, I, D, A, R, T, H, E | Modelo detalhado (Italia) |

## Resultados

Os resultados sao salvos em `results/`:
- `parameters/`: parametros identificados (JSON)
- `figures/`: graficos de ajuste e controle (PNG)

## Dependencias

- numpy >= 1.21.0
- scipy >= 1.7.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- pyyaml >= 5.4.0
