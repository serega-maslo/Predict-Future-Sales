from typing import Protocol

import pandas as pd
import numpy as np


class ModelProtocol(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ModelProtocol": ...
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


def validation_chema(df: pd.DataFrame, model_class: ModelProtocol, loss_function: callable)->pd.Series:
    results = pd.Series()
    max_month = df['date_block_num'].max()
    columns_to_group = ['corrected_shop_id', 'corrected_item_id']
    train = df.iloc[0:0].copy()
    for month in range(max_month):
        train = pd.concat(
            [train, df[df["date_block_num"] == month]],
            ignore_index=True,
        )
        valid = df[df["date_block_num"] == month + 1][['corrected_shop_id', 'corrected_item_id', 'item_cnt_day']]
        valid = (
            valid
            .groupby(columns_to_group, as_index=False)
            .agg(item_cnt_day=("item_cnt_day", "sum"))
        )
        model = model_class()
        model.fit(train)
        true_ans = valid['item_cnt_day']
        valid = valid.drop(columns=['item_cnt_day'])
        prediction = model.predict(valid)
        results.loc[month + 1] = loss_function(true_ans, prediction)
    return results
