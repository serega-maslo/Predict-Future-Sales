import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.constants import INFLATION_DATA

raw_data_path = project_root / "data" / "raw"
cache_path = project_root / "data" / "processed"
cache_path.mkdir(parents=True, exist_ok=True)



def _get_main_category(item_category_name):
    return item_category_name.split('-', 1)[0].strip()

def _clean_item_name(text):
        text = re.sub(r'^[\!\*\/\s]+', '', text)
        text = re.sub(r'[\!\*\/\s]$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+D$', ' Disc', text, flags=re.IGNORECASE)
        return text.strip()


def _compute_item_cats(raw_data_path: Path) -> pd.DataFrame:
    item_cats = pd.read_csv(raw_data_path / "item_categories.csv")
    item_cats["main_category"] = item_cats["item_category_name"].map(_get_main_category)
    return item_cats


def _clean_shop_name(text):
    return text.lstrip('!')


def _get_shop_city(text):
    city = text.split()[0]
    if city == 'Выездная':
        city = 'Выездная торговля'
    elif city == 'Цифровой' or city == 'Интернет-магазин':
        city = 'Интернет'
    return city



def _compute_items(raw_data_path: Path) -> pd.DataFrame:
    items = pd.read_csv(raw_data_path / "items.csv")
    items['item_name'] = items['item_name'].apply(_clean_item_name)
    items['simple_name'] = (
        items['item_name']
        .astype(str)
        .str.replace(r'[^a-zA-Zа-яА-ЯёЁ0-9]', '', regex=True)
        .str.lower()
    )
    corrected = items.drop_duplicates('simple_name').set_index('simple_name')[['item_name', 'item_id']]
    items['corrected_item_id'] = items['simple_name'].map(corrected['item_id'])
    items.drop(columns='simple_name', inplace=True)
    return items



def _compute_shops(raw_data_path: Path) -> pd.DataFrame:
    shops = pd.read_csv(raw_data_path / "shops.csv")
    shops['shop_name'] = shops["shop_name"].apply(_clean_shop_name)
    shops['city'] = shops['shop_name'].apply(_get_shop_city)
    shops['corrected_shop_id'] = shops['shop_id']
    shops.loc[shops['shop_name'] == 'Якутск Орджоникидзе, 56 фран', ['corrected_shop_id']] = [shops[shops["shop_name"] == 'Якутск Орджоникидзе, 56'].iloc[0]['shop_id']]
    shops.loc[shops['shop_name'] == 'Якутск ТЦ "Центральный" фран', ['corrected_shop_id']] = [shops[shops["shop_name"] == 'Якутск ТЦ "Центральный"'].iloc[0]['shop_id']]
    shops.loc[shops['shop_name'] == 'Жуковский ул. Чкалова 39м?', ['corrected_shop_id']] = [shops[shops["shop_name"] == 'Жуковский ул. Чкалова 39м²'].iloc[0]['shop_id']]
    return shops


def _compute_train(raw_data_path: Path) -> pd.DataFrame:
    train = pd.read_csv(raw_data_path / "sales_train.csv")
    train = train.drop_duplicates(keep='first')
    train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
    train["year"] = train["date"].dt.year
    train["month"] = train["date"].dt.month
    train["day"] = train["date"].dt.day
    train["week_day"] = train["date"].dt.day_of_week
    train["month_day_sin"] = np.sin(2 * np.pi * train["day"] / train["date"].dt.days_in_month)
    train["month_day_cos"] = np.cos(2 * np.pi * train["day"] / train["date"].dt.days_in_month)
    train["month_sin"] = np.sin(2 * np.pi * train["month"] / 12)
    train["month_cos"] = np.cos(2 * np.pi * train["month"] / 12)
    train["week_day_sin"] = np.sin(2 * np.pi * train["week_day"] / 7)
    train["week_day_cos"] = np.cos(2 * np.pi * train["week_day"] / 7)
    train['inflation_factor'] = train.set_index(['year', 'month']).index.map(INFLATION_DATA)
    train['no_inflation_price'] = train['item_price'] / train['inflation_factor']
    train.drop(columns=['inflation_factor'], inplace=True)
    return train



def _compute_df_final(raw_data_path: Path) -> pd.DataFrame:
    shops = shops_loader().load()
    items = items_loader().load()
    item_cats = item_category_loader().load()
    train = train_loader().load()

    df_final = (
    train
    .merge(
        items[
            ["item_id", "corrected_item_id", "item_category_id", "item_name"]
        ],
        on="item_id",
        how="left"
    )
    .merge(
        item_cats[
            ["item_category_id", "main_category"]
        ],
        on="item_category_id",
        how="left"
    )
    .merge(
        shops[
            ["shop_id", "corrected_shop_id", "city"]
        ],
        on="shop_id",
        how="left"
    )
    .drop(columns=["item_category_id", "shop_id", "item_id"])
    )
    df_final.loc[484683, ["item_price", "no_inflation_price"]] = [1249.0, 1249.0 / INFLATION_DATA[2013, 5]]
    df_final.loc[1256153, ['item_price', 'item_cnt_day', 'no_inflation_price']] = [307980.0 / 522, 522, 289183.0985915493 / 522]
    indices_to_drop = df_final[df_final["item_name"] == "Доставка (EMS)"].nlargest(1, "item_price").index
    df_final = df_final.drop(indices_to_drop)
    df_final.drop([2909818, 2864235, 2851091, 2608040, 2626181, 2851073, 2067669, 2864260], inplace=True)
    indices_to_drop = df_final[df_final["item_name"] == "Доставка до пункта выдачи (Boxberry)"].nlargest(2, "item_cnt_day").index
    df_final = df_final.drop(indices_to_drop)
    columns_to_drop = ['date', 'month', 'day', 'week_day', 'item_name']
    df_final.drop(columns=columns_to_drop, inplace=True)
    return df_final




def _compute_for_submission(raw_data_path: Path) -> pd.DataFrame:
    for_submission = pd.read_csv(raw_data_path / "test.csv")
    shops = shops_loader().load()
    items = items_loader().load()
    for_submission = (
    for_submission
    .merge(
        items[
            ["item_id", "corrected_item_id"]
        ],
        on="item_id",
        how="left"
    )
    .merge(
        shops[
            ["shop_id", "corrected_shop_id"]
        ],
        on="shop_id",
        how="left"
    )
    )
    return for_submission



class base_loader:
    cache_name: str
    compute_fn: callable


    def __init__(self, raw_data_path: Path = raw_data_path, cache_path: Path = cache_path):
        self.raw_data_path = raw_data_path
        self.cache_path = cache_path

    @property
    def cache_file(self):
        return self.cache_path / f"{self.cache_name}.csv"

    def load(self, recompute=True) -> pd.DataFrame:
        if self.cache_file.exists() and not recompute:
            return pd.read_csv(self.cache_file)

        df = self.compute_fn(self.raw_data_path)
        df.to_csv(self.cache_file, index=False)
        return df
    

class item_category_loader(base_loader):
    cache_name = "item_cats_processed"
    compute_fn = staticmethod(_compute_item_cats)

class items_loader(base_loader):
    cache_name = "items_processed"
    compute_fn = staticmethod(_compute_items)

class shops_loader(base_loader):
    cache_name = "shops_processed"
    compute_fn = staticmethod(_compute_shops)

class train_loader(base_loader):
    cache_name = "train_processed"
    compute_fn = staticmethod(_compute_train)

class df_final_loader(base_loader):
    cache_name = "df_final"
    compute_fn = staticmethod(_compute_df_final)

class for_submission_loader(base_loader):
    cache_name = "for_submission"
    compute_fn = staticmethod(_compute_for_submission)
