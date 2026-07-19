import re
import sys
from pathlib import Path

import pandas as pd

from src.data.constants import INFLATION_DATA
from src.data.items import _compute_items
from src.data.item_cats import _compute_item_cats
from src.data.shops import _compute_shops
from src.data.train import _compute_train
from src.data.df_final import _compute_df_final
from src.data.for_submission import _compute_for_submission


class BaseLoader:
    cache_name: str
    compute_fn: callable


    def __init__(self, raw_data_path: Path, cache_path: Path):
        self.raw_data_path = raw_data_path
        self.cache_path = cache_path

    @property
    def cache_file(self):
        return self.cache_path / f"{self.cache_name}.csv"

    def load(self, recompute=False) -> pd.DataFrame:
        if self.cache_file.exists() and not recompute:
            return pd.read_csv(self.cache_file)

        df = self.compute_fn()
        df.to_csv(self.cache_file, index=False)
        return df
    

class ItemCategoryLoader(BaseLoader):
    cache_name = "item_cats_processed"
    
    def compute_fn(self):
        return _compute_item_cats(self.raw_data_path)

class ItemsLoader(BaseLoader):
    cache_name = "items_processed"
    
    def compute_fn(self):
        return _compute_items(self.raw_data_path)

class ShopsLoader(BaseLoader):
    cache_name = "shops_processed"
    
    def compute_fn(self):
        return _compute_shops(self.raw_data_path)

class TrainLoader(BaseLoader):
    cache_name = "train_processed"
    
    def compute_fn(self):
        return _compute_train(self.raw_data_path)

class DfFinalLoader(BaseLoader):

    def __init__(self, cache_path: Path, shops: pd.DataFrame, items: pd.DataFrame, item_cats: pd.DataFrame, train: pd.DataFrame):
        self.cache_path = cache_path
        self.shops = shops
        self.items = items
        self.item_cats = item_cats
        self.train = train

    cache_name = "df_final"

    def compute_fn(self):
        return _compute_df_final(
            self.shops,
            self.items,
            self.item_cats,
            self.train,
        )
    
    

class ForSubmissionLoader(BaseLoader):

    def __init__(self, raw_data_path: Path, cache_path: Path, shops: pd.DataFrame, items: pd.DataFrame):
        self.raw_data_path = raw_data_path
        self.cache_path = cache_path
        self.shops = shops
        self.items = items
    cache_name = "for_submission"

    def compute_fn(self):
        return _compute_for_submission(
            self.raw_data_path,
            self.shops,
            self.items,
        )
