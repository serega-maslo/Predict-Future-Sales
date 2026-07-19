import numpy as np
import pandas as pd

def _get_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["month_day_sin"] = np.sin(2 * np.pi * df["day"] / df["date"].dt.days_in_month)
    df["month_day_cos"] = np.cos(2 * np.pi * df["day"] / df["date"].dt.days_in_month)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_day_sin"] = np.sin(2 * np.pi * df["week_day"] / 7)
    df["week_day_cos"] = np.cos(2 * np.pi * df["week_day"] / 7)
    df.drop(columns=['date', 'month', 'day', 'week_day'])
    return df

def _get_shops_speciality(df: pd.DataFrame) -> pd.DataFrame:
    shop_speciality = (
    df
    .groupby(["shop_name", "main_category"], as_index=False)["item_cnt_day"]
    .sum()
    .sort_values(["shop_name", "item_cnt_day"], ascending=[True, False])
    .drop_duplicates("shop_name")
    .rename(columns={"main_category": "shop_speciality"})
    [["shop_name", "shop_speciality"]]
)

    df = df.merge(shop_speciality, on="shop_name", how="left")
    return df


def _get_item_is_shops_speciality(df: pd.DataFrame) -> pd.DataFrame:
    df["item_is_shops_speciality"] = (df['shop_speciality'] == df['main_category']).astype(int)
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _get_date_features(df)
    df = _get_shops_speciality(df)
    df = _get_item_is_shops_speciality(df)
    return df