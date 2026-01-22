"""
Utilitários para Carregamento e Preparação de Dados COVID-19

Este módulo contém funções para:
- Carregar dados COVID-19 do Brasil de múltiplos arquivos CSV
- Preparar compartimentos SIR (Suscetíveis, Infectados, Removidos)
- Suavizar dados usando média móvel
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict


def load_covid_data_brazil(
    data_dir: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Carrega dados COVID-19 do Brasil de múltiplos arquivos CSV

    Parâmetros
    ----------
    data_dir : str
        Diretório contendo os arquivos CSV
    start_date : str, optional
        Data de início no formato 'YYYY-MM-DD'
    end_date : str, optional
        Data de fim no formato 'YYYY-MM-DD'

    Retorna
    -------
    pd.DataFrame
        DataFrame com dados COVID-19 filtrados para nível nacional
        Colunas: data, casosAcumulado, obitosAcumulado, Recuperadosnovos, populacaoTCU2019
    """
    data_path = Path(data_dir)

    # Encontrar todos os arquivos CSV de dados COVID
    csv_files = sorted(data_path.glob("HIST_PAINEL_COVIDBR_*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {data_dir}")

    print(f"Encontrados {len(csv_files)} arquivos CSV")

    # Lista para armazenar DataFrames
    dfs = []

    for csv_file in csv_files:
        print(f"Carregando {csv_file.name}...")

        # Ler CSV com separador ponto-e-vírgula (padrão brasileiro)
        df = pd.read_csv(
            csv_file,
            sep=';',
            parse_dates=['data'],
            encoding='utf-8',
            low_memory=False
        )

        # Filtrar apenas dados do Brasil (nível nacional)
        df_brasil = df[df['regiao'] == 'Brasil'].copy()

        if len(df_brasil) > 0:
            dfs.append(df_brasil)

    # Concatenar todos os DataFrames
    df_completo = pd.concat(dfs, ignore_index=True)

    # Ordenar por data
    df_completo = df_completo.sort_values('data').reset_index(drop=True)

    # Remover duplicatas (manter a última ocorrência)
    df_completo = df_completo.drop_duplicates(subset=['data'], keep='last')

    # Filtrar por intervalo de datas se especificado
    if start_date:
        df_completo = df_completo[df_completo['data'] >= start_date]

    if end_date:
        df_completo = df_completo[df_completo['data'] <= end_date]

    # Selecionar colunas relevantes
    colunas_relevantes = [
        'data',
        'casosAcumulado',
        'obitosAcumulado',
        'Recuperadosnovos',
        'populacaoTCU2019'
    ]

    # Verificar se todas as colunas existem
    colunas_disponiveis = [col for col in colunas_relevantes if col in df_completo.columns]

    df_final = df_completo[colunas_disponiveis].copy()

    # Converter para numérico, substituindo erros por NaN
    for col in colunas_disponiveis:
        if col != 'data':
            df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    # Preencher valores faltantes com interpolação linear
    df_final = df_final.interpolate(method='linear', limit_direction='both')

    # Preencher qualquer NaN restante com 0
    df_final = df_final.fillna(0)

    print(f"Dados carregados: {len(df_final)} registros de {df_final['data'].min()} a {df_final['data'].max()}")

    return df_final


def smooth_data(data: np.ndarray, window: int = 7) -> np.ndarray:
    """
    Aplica média móvel para suavizar dados ruidosos

    Parâmetros
    ----------
    data : np.ndarray
        Array de dados a ser suavizado
    window : int, default=7
        Tamanho da janela para média móvel

    Retorna
    -------
    np.ndarray
        Dados suavizados
    """
    if len(data) < window:
        return data

    # Usar convolução para média móvel
    weights = np.ones(window) / window
    smoothed = np.convolve(data, weights, mode='same')

    # Ajustar bordas (primeiros e últimos elementos)
    for i in range(window // 2):
        smoothed[i] = np.mean(data[:i+window//2+1])
        smoothed[-(i+1)] = np.mean(data[-(i+window//2+1):])

    return smoothed


def prepare_sir_data(
    df: pd.DataFrame,
    population: float,
    smooth: bool = True,
    window: int = 7
) -> Dict[str, np.ndarray]:
    """
    Converte dados COVID-19 para compartimentos SIR

    A lógica de conversão:
    - R(t) = Recuperadosnovos.cumsum() + obitosAcumulado (removidos da transmissão)
    - I(t) = casosAcumulado - R(t) (infectados ativos)
    - S(t) = N - I(t) - R(t) (suscetíveis)

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com dados COVID-19
    population : float
        População total
    smooth : bool, default=True
        Se True, aplica suavização nos dados
    window : int, default=7
        Tamanho da janela para média móvel

    Retorna
    -------
    dict
        Dicionário com:
        - 'time': np.ndarray - Dias desde o início (0, 1, 2, ...)
        - 'dates': pd.DatetimeIndex - Datas correspondentes
        - 'S': np.ndarray - Suscetíveis
        - 'I': np.ndarray - Infectados
        - 'R': np.ndarray - Removidos
        - 'population': float - População total
    """
    # Calcular compartimentos

    # Recuperados acumulados (somar novos recuperados ao longo do tempo)
    if 'Recuperadosnovos' in df.columns:
        recuperados_acumulado = df['Recuperadosnovos'].cumsum().values
    else:
        # Se não houver dados de recuperados, usar estimativa
        # Assumir que recuperados = casos acumulados - óbitos - casos ativos estimados
        recuperados_acumulado = np.zeros(len(df))

    # Óbitos acumulados
    obitos_acumulado = df['obitosAcumulado'].values

    # Casos acumulados
    casos_acumulado = df['casosAcumulado'].values

    # R(t) = Recuperados + Óbitos (removidos da transmissão)
    R = recuperados_acumulado + obitos_acumulado

    # I(t) = Casos acumulados - Removidos
    I = casos_acumulado - R

    # Garantir que I não seja negativo
    I = np.maximum(I, 0)

    # S(t) = População - I(t) - R(t)
    S = population - I - R

    # Garantir que S não seja negativo
    S = np.maximum(S, 0)

    # Aplicar suavização se solicitado
    if smooth:
        S = smooth_data(S, window)
        I = smooth_data(I, window)
        R = smooth_data(R, window)

    # Validação: S + I + R deve estar próximo de N
    total = S + I + R
    discrepancia = np.abs(total - population)
    max_discrepancia = np.max(discrepancia)

    if max_discrepancia > population * 0.01:  # Mais de 1% de erro
        print(f"Aviso: Discrepância máxima de {max_discrepancia:,.0f} "
              f"({max_discrepancia/population*100:.2f}%) na soma S+I+R")

    # Criar array de tempo (dias desde o início)
    time = np.arange(len(df))

    # Datas correspondentes
    dates = pd.DatetimeIndex(df['data'])

    return {
        'time': time,
        'dates': dates,
        'S': S,
        'I': I,
        'R': R,
        'population': population
    }


def prepare_seir_data(
    df: pd.DataFrame,
    population: float,
    smooth: bool = True,
    window: int = 7
) -> Dict[str, np.ndarray]:
    """
    Converte dados COVID-19 para compartimentos SEIR

    A lógica de conversão:
    - R(t) = Recuperadosnovos.cumsum() + obitosAcumulado (removidos da transmissão)
    - I(t) = casosAcumulado - R(t) (infectados ativos)
    - E(t) = 0 (expostos - não observável diretamente, será estimado pelo modelo)
    - S(t) = N - E(t) - I(t) - R(t) (suscetíveis)

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com dados COVID-19
    population : float
        População total
    smooth : bool, default=True
        Se True, aplica suavização nos dados
    window : int, default=7
        Tamanho da janela para média móvel

    Retorna
    -------
    dict
        Dicionário com:
        - 'time': np.ndarray - Dias desde o início (0, 1, 2, ...)
        - 'dates': pd.DatetimeIndex - Datas correspondentes
        - 'S': np.ndarray - Suscetíveis
        - 'E': np.ndarray - Expostos (inicializado em 0)
        - 'I': np.ndarray - Infectados
        - 'R': np.ndarray - Removidos
        - 'population': float - População total
    """
    # Calcular compartimentos I e R (mesma lógica do SIR)

    # Recuperados acumulados (somar novos recuperados ao longo do tempo)
    if 'Recuperadosnovos' in df.columns:
        recuperados_acumulado = df['Recuperadosnovos'].cumsum().values
    else:
        # Se não houver dados de recuperados, usar estimativa
        recuperados_acumulado = np.zeros(len(df))

    # Óbitos acumulados
    obitos_acumulado = df['obitosAcumulado'].values

    # Casos acumulados
    casos_acumulado = df['casosAcumulado'].values

    # R(t) = Recuperados + Óbitos (removidos da transmissão)
    R = recuperados_acumulado + obitos_acumulado

    # I(t) = Casos acumulados - Removidos
    I = casos_acumulado - R

    # Garantir que I não seja negativo
    I = np.maximum(I, 0)

    # E(t) = 0 (expostos - não observável, será estimado na otimização)
    # Inicializamos com zeros pois não temos dados diretos de expostos
    E = np.zeros_like(I)

    # S(t) = População - E(t) - I(t) - R(t)
    S = population - E - I - R

    # Garantir que S não seja negativo
    S = np.maximum(S, 0)

    # Aplicar suavização se solicitado
    if smooth:
        S = smooth_data(S, window)
        E = smooth_data(E, window)
        I = smooth_data(I, window)
        R = smooth_data(R, window)

    # Validação: S + E + I + R deve estar próximo de N
    total = S + E + I + R
    discrepancia = np.abs(total - population)
    max_discrepancia = np.max(discrepancia)

    if max_discrepancia > population * 0.01:  # Mais de 1% de erro
        print(f"Aviso: Discrepância máxima de {max_discrepancia:,.0f} "
              f"({max_discrepancia/population*100:.2f}%) na soma S+E+I+R")

    # Criar array de tempo (dias desde o início)
    time = np.arange(len(df))

    # Datas correspondentes
    dates = pd.DatetimeIndex(df['data'])

    return {
        'time': time,
        'dates': dates,
        'S': S,
        'E': E,
        'I': I,
        'R': R,
        'population': population
    }


def prepare_sidarthe_data(
    df: pd.DataFrame,
    smooth: bool = True,
    window: int = 7
) -> pd.DataFrame:
    """
    Converte dados COVID-19 para formato adequado ao modelo SIDARTHE

    O modelo SIDARTHE precisa de:
    - confirmed: Casos confirmados totais (D + R + T + H + E)
    - active: Casos ativos (D + R + T)
    - deaths: Mortes acumuladas (E)
    - recovered: Recuperados (H)

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com dados COVID-19 brutos
    smooth : bool, default=True
        Se True, aplica suavização nos dados
    window : int, default=7
        Tamanho da janela para média móvel

    Retorna
    -------
    pd.DataFrame
        DataFrame com colunas: date, confirmed, deaths, recovered, active
    """
    # Extrair dados brutos
    dates = pd.DatetimeIndex(df['data'])
    casos_acumulado = df['casosAcumulado'].values
    obitos_acumulado = df['obitosAcumulado'].values

    # Recuperados acumulados
    # IMPORTANTE: 'Recuperadosnovos' já é acumulado apesar do nome!
    if 'Recuperadosnovos' in df.columns:
        recuperados_acumulado = df['Recuperadosnovos'].values
    else:
        # Estimativa: 80% dos casos confirmados não fatais se recuperam
        # após 14 dias (aproximação grosseira)
        recuperados_acumulado = np.zeros(len(df))
        for i in range(14, len(df)):
            casos_14_dias_atras = casos_acumulado[i-14]
            obitos_ate_agora = obitos_acumulado[i]
            recuperados_acumulado[i] = max(0, casos_14_dias_atras * 0.8 - obitos_ate_agora)

    # Casos ativos = Casos acumulados - Recuperados - Óbitos
    ativos = casos_acumulado - recuperados_acumulado - obitos_acumulado
    ativos = np.maximum(ativos, 0)  # Garantir não-negatividade

    # Aplicar suavização se solicitado
    if smooth:
        casos_acumulado = smooth_data(casos_acumulado, window)
        obitos_acumulado = smooth_data(obitos_acumulado, window)
        recuperados_acumulado = smooth_data(recuperados_acumulado, window)
        ativos = smooth_data(ativos, window)

    # Criar DataFrame no formato esperado pelo SIDARTHE
    df_sidarthe = pd.DataFrame({
        'date': dates,
        'confirmed': casos_acumulado,
        'deaths': obitos_acumulado,
        'recovered': recuperados_acumulado,
        'active': ativos
    })

    return df_sidarthe
