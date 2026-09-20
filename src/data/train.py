from pathlib import Path

import numpy as np
import pandas as pd

from src.data.constants import INFLATION_DATA


def _compute_train(raw_data_path: Path) -> pd.DataFrame:
    train = pd.read_csv(raw_data_path / "sales_train.csv")
    train = train.drop_duplicates(keep='first')
    train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
    train["year"] = train["date"].dt.year
    train["month"] = train["date"].dt.month
    train["day"] = train["date"].dt.day
    train["week_day"] = train["date"].dt.day_of_week
    train['inflation_factor'] = train.set_index(['year', 'month']).index.map(INFLATION_DATA)
    train['no_inflation_price'] = train['item_price'] / train['inflation_factor']
    train.drop(columns=['inflation_factor'], inplace=True)
    return train
