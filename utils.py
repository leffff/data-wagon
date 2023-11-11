import pandas as pd
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
