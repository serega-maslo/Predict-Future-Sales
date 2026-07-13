import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

raw_data_path = project_root / "data" / "raw"
cache_path = project_root / "data" / "processed"
cache_path.mkdir(parents=True, exist_ok=True)

'''from src.data.constants import (
    CLUSTERS_OF_ITEMS,
    INFLATION_DATA,
    NUMBER_OF_DAYS_IN_MONTH,
    NUMBER_OF_DAYS_OF_MONTH,
    NUMBER_OF_DAYS_OF_WEEK,
    POPULATION_DATA,
)'''



def _cache_file(name: str) -> Path:
    return cache_path / f"{name}.csv"

def _load_or_compute(cache_name, recompute, compute_fn):
    cache = _cache_file(cache_name)

    if cache.exists() and not recompute:
        return pd.read_csv(cache)

    df = compute_fn()
    df.to_csv(cache, index=False)
    return df


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



def load_item_cats(
    raw_data_path=raw_data_path,
    recompute=True,
):
    return _load_or_compute(
        cache_name="item_cats_processed",
        recompute=recompute,
        compute_fn=lambda: _compute_item_cats(raw_data_path),
    ) 


def load_items(
    raw_data_path=raw_data_path,
    recompute=True,
):
    return _load_or_compute(
        cache_name="shops_processed",
        recompute=recompute,
        compute_fn=lambda: _compute_items(raw_data_path),
    ) 

if __name__ == "__main__":
    data = load_items(recompute=True)
    print(data.head())





'''
def load_train(
    raw_data_path=raw_data_path,
    recompute=True,
    split="all",          # "all", "train", "val", "test"
    val_block=32,
    test_block=33,
):
    def compute():

        train = pd.read_csv(raw_data_path / "sales_train.csv")
        test = pd.read_csv(raw_data_path / "test.csv")
        items = pd.read_csv(raw_data_path / "items.csv")
        item_cats = pd.read_csv(raw_data_path / "item_categories.csv")
        shops = pd.read_csv(raw_data_path / "shops.csv")



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


        def clean_name(text):
            text = re.sub(r'^[\!\*\/\s]+', '', text)
            text = re.sub(r'[\!\*\/\s]$', '', text, flags=re.IGNORECASE)
            return text.strip()
        items['item_name'] = items['item_name'].apply(clean_name)
        sorted(items["item_name"].unique())


        items['item_name'] = items['item_name'].str.replace(r'\s+D$', ' Disc', regex=True, case=False)
        sorted(items[items['item_name'].str.contains(r' +D$', case=True, na=False)]["item_name"].unique())

        items['cleaned_name'] = (
            items['item_name']
            .astype(str)
            .str.replace(r'[^a-zA-Zа-яА-ЯёЁ0-9]', '', regex=True)
            .str.lower()
        )

        grouped = items.groupby('cleaned_name')['item_name'].apply(list)

        duplicates = grouped[grouped.apply(len) > 1]
        corrected_item_id_map = {}

        for _, original_list in duplicates.items():
            first_item = original_list[0]
            first_id = items[items["item_name"] == first_item].iloc[0]["item_id"]
            for item_name in original_list:
                item_ids = items[items["item_name"] == item_name]["item_id"].tolist()
                for item_id in item_ids:
                    corrected_item_id_map[item_id] = first_id
                items.loc[items['item_name'] == item_name, 'item_name'] = first_item
        

        train["item_id"] = train["item_id"].replace(corrected_item_id_map)


        group_cols = train.columns.drop('item_cnt_day').tolist()
        train = train.groupby(group_cols, as_index=False)['item_cnt_day'].sum()


        split_cats = item_cats['item_category_name'].str.split('-', n=1, expand=True)

        item_cats['main_category'] = split_cats[0].str.strip()

        if split_cats.shape[1] > 1:
            item_cats['subcategory'] = split_cats[1].str.strip()
            item_cats['subcategory'] = item_cats['subcategory'].fillna(item_cats['main_category'])
        else:
            item_cats['subcategory'] = item_cats['main_category']

        item_cats[['item_category_name', 'main_category', 'subcategory']].head(15)
        item_cats["cluster"] = (
        item_cats["item_category_name"]
        .map(CLUSTERS_OF_ITEMS)
        .fillna(9)
        .astype(int)
)


        shops['shop_name'] = shops['shop_name'].str.lstrip('!')
        shops["city"] = shops["shop_name"].str.split().str[0]
        shops['city'] = shops['city'].replace({
            'Выездная': 'Выездная торговля',
            'Цифровой': 'Интернет',
            'Интернет-магазин': 'Интернет'
        })

        corrected_shop_id_map = {}
        corrected_shop_id_map[shops[shops["shop_name"] == 'Якутск Орджоникидзе, 56 фран'].iloc[0]["shop_id"]] = shops[shops["shop_name"] == 'Якутск Орджоникидзе, 56'].iloc[0]["shop_id"]
        corrected_shop_id_map[shops[shops["shop_name"] == 'Якутск ТЦ "Центральный" фран'].iloc[0]["shop_id"]] = shops[shops["shop_name"] == 'Якутск ТЦ "Центральный"'].iloc[0]["shop_id"]
        corrected_shop_id_map[shops[shops["shop_name"] == 'Жуковский ул. Чкалова 39м?'].iloc[0]["shop_id"]] = shops[shops["shop_name"] == 'Жуковский ул. Чкалова 39м²'].iloc[0]["shop_id"]
        train["shop_id"] = train["shop_id"].replace(corrected_shop_id_map)
        test["shop_id"] = test["shop_id"].replace(corrected_shop_id_map)


        group_cols = train.columns.drop('item_cnt_day').tolist()
        train = train.groupby(group_cols, as_index=False)['item_cnt_day'].sum()

        items_completed = pd.merge(items, item_cats, on="item_category_id", how="left")
        train_merged = pd.merge(train, items_completed, on="item_id", how="left")
        df_final = pd.merge(train_merged, shops, on="shop_id", how="left")

        df_final.loc[484683, "item_price"] = 1249.0
        df_final.loc[484683, "no_inflation_price"] = 1249.0 / INFLATION_DATA[2013, 5]

        df_final.loc[df_final['item_name'] == 'Radmin 3  - 522 лиц.', ['item_name', 'item_price', 'item_cnt_day', 'no_inflation_price']] = ["Radmin 3", 307980.0 / 522, 522, 289183.0985915493 / 522]



        indices_to_drop = df_final[df_final["item_name"] == "Доставка (EMS)"].nlargest(1, "item_price").index
        df_final = df_final.drop(indices_to_drop)


        df_final.drop([2909818, 2864235, 2851091, 2608040, 2626181, 2851073, 2067669, 2864260], inplace=True)

        indices_to_drop = df_final[df_final["item_name"] == "Доставка до пункта выдачи (Boxberry)"].nlargest(2, "item_cnt_day").index
        df_final = df_final.drop(indices_to_drop)
        

        shop_speciality_sales = (
            df_final.groupby(['shop_name', 'main_category'])['item_cnt_day']
            .sum()
            .reset_index()
        ) 
        shop_speciality_sales = shop_speciality_sales.sort_values(['shop_name', 'item_cnt_day'], ascending=[True, False])
        shop_speciality_sales['rank'] = shop_speciality_sales.groupby('shop_name').cumcount() + 1
        top_category = shop_speciality_sales[shop_speciality_sales['rank'] <= 1]
        shop_specs = (
            top_category.pivot(index='shop_name', columns='rank', values='main_category')
            .rename(columns={1: 'shop_speciality'})
            .fillna('Нет')
        )
        df_final = df_final.merge(shop_specs, on='shop_name', how='left')
        columns_to_drop = ['date', 'month', 'day', 'week_day', 'inflation_factor', 'item_name', 'cleaned_name', 'item_category_name', 'item_category_id', 'shop_name', 'main_category', 'subcategory']
        df_final.drop(columns=columns_to_drop, inplace=True)
        return df_final
    
    df = _load_or_compute(
        cache_name="train_processed",
        recompute=recompute,
        compute_fn=compute,
    )

    if split == "train":
        return df[df["date_block_num"] < val_block].reset_index(drop=True)

    elif split == "val":
        return df[df["date_block_num"] == val_block].reset_index(drop=True)

    elif split == "test":
        return df[df["date_block_num"] == test_block].reset_index(drop=True)

    elif split == "all":
        return df

    else:
        raise ValueError(
            "split must be one of {'all', 'train', 'val', 'test'}"
        )




def load_for_submission(
        raw_data_path=raw_data_path,
        recompute=True,
):
    def compute():
        for_submission = pd.read_csv(raw_data_path / "test.csv")
        items = pd.read_csv(raw_data_path / "items.csv")
        shops = pd.read_csv(raw_data_path / "shops.csv")



        def clean_name(text):
            text = re.sub(r'^[\!\*\/\s]+', '', text)
            text = re.sub(r'[\!\*\/\s]$', '', text, flags=re.IGNORECASE)
            return text.strip()
        items['item_name'] = items['item_name'].apply(clean_name)
        sorted(items["item_name"].unique())


        items['item_name'] = items['item_name'].str.replace(r'\s+D$', ' Disc', regex=True, case=False)
        sorted(items[items['item_name'].str.contains(r' +D$', case=True, na=False)]["item_name"].unique())



        items['cleaned_name'] = (
            items['item_name']
            .astype(str)
            .str.replace(r'[^a-zA-Zа-яА-ЯёЁ0-9]', '', regex=True)
            .str.lower()
        )

        grouped = items.groupby('cleaned_name')['item_name'].apply(list)

        duplicates = grouped[grouped.apply(len) > 1]
        corrected_item_id_map = {}

        for _, original_list in duplicates.items():
            first_item = original_list[0]
            first_id = items[items["item_name"] == first_item].iloc[0]["item_id"]
            for item_name in original_list:
                item_ids = items[items["item_name"] == item_name]["item_id"].tolist()
                for item_id in item_ids:
                    corrected_item_id_map[item_id] = first_id
                items.loc[items['item_name'] == item_name, 'item_name'] = first_item
        
        for_submission["item_id"] = for_submission["item_id"].replace(corrected_item_id_map)


        shops['shop_name'] = shops['shop_name'].str.lstrip('!')
        corrected_shop_id_map = {}
        corrected_shop_id_map[shops[shops["shop_name"] == 'Якутск Орджоникидзе, 56 фран'].iloc[0]["shop_id"]] = shops[shops["shop_name"] == 'Якутск Орджоникидзе, 56'].iloc[0]["shop_id"]
        corrected_shop_id_map[shops[shops["shop_name"] == 'Якутск ТЦ "Центральный" фран'].iloc[0]["shop_id"]] = shops[shops["shop_name"] == 'Якутск ТЦ "Центральный"'].iloc[0]["shop_id"]
        corrected_shop_id_map[shops[shops["shop_name"] == 'Жуковский ул. Чкалова 39м?'].iloc[0]["shop_id"]] = shops[shops["shop_name"] == 'Жуковский ул. Чкалова 39м²'].iloc[0]["shop_id"]
        for_submission["shop_id"] = for_submission["shop_id"].replace(corrected_shop_id_map)
        return for_submission


    
    return _load_or_compute(
        cache_name="for_submission_processed",
        recompute=recompute,
        compute_fn=compute,
    )


'''
