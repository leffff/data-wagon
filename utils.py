import pandas as pd
import numpy as np

import datetime

from typing import List


def train_test_split(data: pd.DataFrame, n_periods_split: int = 5, group_columns: List[str] = None) -> pd.DataFrame:
    if group_columns is None:
        group_columns = ["client_sap_id", "freight_id", "sender_station_id",
                         "recipient_station_id", "sender_organisation_id"]

    data_grouped = data.groupby(group_columns).agg(list)

    def __train_test_split(periods):
        if len(periods) >= n_periods_split:
            return [False] * (len(periods) - 2) + [True] * 2
        else:
            return [False] * len(periods)

    data_grouped["is_test"] = data_grouped["period"].apply(__train_test_split)
    explode_columns = data_grouped.columns.tolist()
    train_test = data_grouped.explode(explode_columns).reset_index()

    return train_test


def get_normal_weight(df, threshold=100):
    cur_df = df.copy()
    solo_wagons = cur_df[cur_df['real_wagon_count'] == 1][['rps', 'podrod', 'real_weight', 'real_wagon_count']]
    solo_wagons = solo_wagons[solo_wagons['real_weight'] < threshold]
    weight_dict = solo_wagons.groupby(['rps', 'podrod'])['real_weight'].agg(
        [('mean_weight', 'mean'), ('q95_weight', lambda x: np.quantile(a=x, q=0.95))]).reset_index()
    cur_df = cur_df.loc[solo_wagons.index]
    cur_df = cur_df.merge(weight_dict, how='left', on=['rps', 'podrod'])
    return cur_df, weight_dict


def add_time_series_features(df, data_cols: str = None):
    if data_cols is None:
        data_cols = ["period"]
    features = []
    for col in data_cols:
        features.extend([col + "_year", col + "_day", col + "_weekday", col + "_month"])
        df[col] = df[col].fillna(df[col].mode())
        df[col] = pd.to_datetime(df[col], format="%Y-%m-%d", errors='coerce')
        df[col + "_year"] = df[col].dt.year
        df[col + "_day"] = df[col].dt.day
        df[col + "_weekday"] = df[col].dt.weekday
        df[col + "_month"] = df[col].dt.month
        df[col + "_seconds"] = df[col].apply(lambda x: (x - datetime.datetime(1970, 1, 1)).total_seconds())
    return df, features


def add_master_data_mappings(df: pd.DataFrame) -> pd.DataFrame:
    client_mapping_file = "./data/client_mapping.csv"
    freight_mapping_file = "./data/freight_mapping.csv"
    station_mapping_file = "./data/station_mapping.csv"
    client_mapping = pd.read_csv(
        client_mapping_file,
        sep=";",
        decimal=",",
        encoding="windows-1251",
    )
    df = pd.merge(df, client_mapping, how="left", on="client_sap_id")
    freight_mapping = pd.read_csv(
        freight_mapping_file, sep=";", decimal=",", encoding="windows-1251"
    )
    df = pd.merge(df, freight_mapping, how="left", on="freight_id")
    station_mapping = pd.read_csv(
        station_mapping_file,
        sep=";",
        decimal=",",
        encoding="windows-1251",
    )
    df = pd.merge(
        df,
        station_mapping.add_prefix("sender_"),
        how="left",
        on="sender_station_id",
    )
    df = pd.merge(
        df,
        station_mapping.add_prefix("recipient_"),
        how="left",
        on="recipient_station_id",
    )

    return df


def evaluate(fact: pd.DataFrame, forecast: pd.DataFrame, public: bool = True) -> float:
    # = Параметры для расчета метрики =
    accuracy_granularity = [
        "period",
        "rps",
        "holding_name",
        "sender_department_name",
        "recipient_department_name",
    ]
    fact_value, forecast_value = "real_wagon_count", "forecast_wagon_count"
    if public:
        metric_weight = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    else:
        metric_weight = np.array([0.1, 0.6, 0.1, 0.1, 0.1])

    # = Собственно расчет метрик =
    # 1. Добавляем сущности верхних уровней гранулярности по справочникам
    # fact = add_master_data_mappings(fact)
    # forecast = add_master_data_mappings(forecast)

    # 2. Расчет KPI
    compare_data = pd.merge(
        fact.groupby(accuracy_granularity, as_index=False)[fact_value].sum(),
        forecast.groupby(accuracy_granularity, as_index=False)[forecast_value].sum(),
        how="outer",
        on=accuracy_granularity,
    ).fillna(0)
    # Против самых хитрых - нецелочисленный прогноз вагоноотправок не принимаем
    compare_data[fact_value] = np.around(compare_data[fact_value]).astype(int)
    compare_data[forecast_value] = np.around(compare_data[forecast_value]).astype(int)

    # 3. Рассчитаем метрики для каждого месяца в выборке
    compare_data["ABS_ERR"] = abs(
        compare_data[forecast_value] - compare_data[fact_value]
    )
    compare_data["MAX"] = abs(compare_data[[forecast_value, fact_value]].max(axis=1))
    summary = compare_data.groupby("period")[
        [forecast_value, fact_value, "ABS_ERR", "MAX"]
    ].sum()
    summary["Forecast Accuracy"] = 1 - summary["ABS_ERR"] / summary["MAX"]

    # 4. Взвесим метрики отдельных месяцев для получения одной цифры score
    score = (
            summary["Forecast Accuracy"].sort_index(ascending=True) * metric_weight
    ).sum()

    return score
