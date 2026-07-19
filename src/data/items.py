from pathlib import Path

import numpy as np
import pandas as pd

from src.data.compute_utils import _clean_item_name


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
